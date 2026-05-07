import argparse
import json
import os
import shutil
import sys

import numpy as np
import PIL
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from diffusers.models import AutoencoderKL

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
sys.path.append("model_repositories/DiT")
sys.path.append(".")

from diffusion import create_diffusion
from models import DiT_models

import sys
sys.path.append('/home/iasudakov/')

from yt_tools.nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot

from image_preprocessing import (
    center_crop_arr,
    random_crop_arr,
)


def _checkpoint_path(checkpoint_dir, rank):
    return os.path.join(checkpoint_dir, f"checkpoint_rank{rank}.pt")


def save_checkpoint(checkpoint_dir, rank, state):
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_path = _checkpoint_path(checkpoint_dir, rank)
    tmp_path = final_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)


def load_checkpoint_if_consistent(checkpoint_dir, rank, world_size, device):
    """Load checkpoint only if every rank has a checkpoint file with matching world_size."""
    path = _checkpoint_path(checkpoint_dir, rank)
    has_local = os.path.exists(path)
    flag = torch.tensor([1 if has_local else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        return None
    ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("world_size") != world_size:
        if rank == 0:
            print(
                f"Checkpoint world_size={ckpt.get('world_size')} does not match "
                f"current world_size={world_size}; ignoring checkpoint."
            )
        return None
    return ckpt


def topk_recall(scores, truth_mask, ks):
    """Mean per-image top-k recall: hits-in-top-k / k, averaged over images with k > 0."""
    accs = []
    for i in range(scores.shape[0]):
        k = int(ks[i].item())
        if k == 0:
            continue
        topk = scores[i].topk(k).indices
        hits = truth_mask[i, topk].sum().float().item()
        accs.append(hits / k)
    return float(np.mean(accs)) if accs else 0.0


def calc_statistics_coco(likelyhoods, truth_mask, ks, n_trials_acc):
    len_dataset, n_trials, n_class = likelyhoods.shape
    n_averaging = n_trials // n_trials_acc
    accs0, accs1, accs2 = [], [], []
    for i in range(n_averaging):
        sl = likelyhoods[:, i * n_trials_acc : (i + 1) * n_trials_acc]
        accs0.append(topk_recall(sl.mean(dim=1), truth_mask, ks))
        accs1.append(topk_recall(torch.softmax(sl, dim=-1).mean(dim=1), truth_mask, ks))
        accs2.append(topk_recall(torch.logsumexp(sl, dim=1), truth_mask, ks))
    return np.array(accs0), np.array(accs1), np.array(accs2)


@torch.no_grad()
def calculate_likelihoods(
    rank, vae, model, diffusion, items, images_dir, classes, batch_size, step_size, g,
    checkpoint_dir=None,
    checkpoint_interval=1,
):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    device = model.y_embedder.embedding_table.weight.device

    n_class = len(classes)
    n_trials = 1000 // step_size

    len_dataset = len(items)

    YS = classes
    assert n_class % (batch_size * dist.get_world_size()) == 0
    batch_iters = n_class // (batch_size * dist.get_world_size())

    ts = np.arange(step_size // 2, 1000, step_size)

    likelyhoods = torch.zeros((len_dataset, n_trials, n_class)).to(device)

    truth_mask = torch.zeros((len_dataset, n_class), dtype=torch.bool, device=device)
    ks = torch.zeros(len_dataset, dtype=torch.long, device=device)
    for img_idx, (_, imagenet_idx) in enumerate(items):
        if len(imagenet_idx) > 0:
            idx_tensor = torch.tensor(imagenet_idx, dtype=torch.long, device=device)
            truth_mask[img_idx].index_fill_(0, idx_tensor, True)
        ks[img_idx] = len(imagenet_idx)

    skip_until = 0
    cnt = 0
    running_hits = 0.0

    if checkpoint_dir is not None:
        ckpt = load_checkpoint_if_consistent(
            checkpoint_dir, rank, dist.get_world_size(), device
        )
        if ckpt is not None:
            if ckpt["len_dataset"] != len_dataset or ckpt["n_class"] != n_class:
                if rank == 0:
                    print(
                        "Checkpoint shape does not match current dataset/classes; "
                        "ignoring checkpoint."
                    )
            else:
                likelyhoods.copy_(ckpt["likelyhoods"].to(device))
                skip_until = int(ckpt["img_num"])
                cnt = int(ckpt.get("cnt", 0))
                running_hits = float(ckpt.get("running_hits", 0.0))
                g.set_state(ckpt["g_state"])
                if rank == 0:
                    print(
                        f"Resuming from checkpoint at img_num={skip_until} "
                        f"(cnt={cnt}, running_hits={running_hits:.4f})"
                    )
        dist.barrier()

    for img_num, (filename, imagenet_idx) in enumerate(items):
        if img_num < skip_until:
            continue

        if len(imagenet_idx) == 0:
            continue

        file_path = os.path.join(images_dir, filename)

        local_likelyhoods = torch.zeros(n_class).to(device)

        for trial, t in enumerate(ts):

            orig_img = PIL.Image.open(file_path).convert("RGB")

            if args.use_augmentations:
                img = random_crop_arr(orig_img, args.image_size, g=g)
            else:
                img = center_crop_arr(orig_img, args.image_size)

            img = transform(img).to(device).unsqueeze(0)
            latents = vae.encode(img).latent_dist.sample().mul_(0.18215)

            batch_ts = torch.tensor([t]).long().to(device)
            noise = torch.randn(latents.shape, generator=g).to(device)
            noised_latents = diffusion.q_sample(latents, batch_ts, noise)
            noised_latents = noised_latents.tile((batch_size, 1, 1, 1))

            for batch_iter in range(batch_iters):
                batch_ind = batch_iter + rank * batch_iters
                cond = YS[batch_ind * batch_size : (batch_ind + 1) * batch_size]

                with torch.autocast(device.type, torch.bfloat16):
                    model_output = model(noised_latents, batch_ts, y=cond)
                B, C = noised_latents.shape[:2]
                noise_pred, _ = torch.split(model_output, C, dim=1)

                loss = ((noise - noise_pred) ** 2).sum(dim=(1, 2, 3))

                likelyhoods[img_num, trial, batch_ind * batch_size : (batch_ind + 1) * batch_size] = -loss
                local_likelyhoods[batch_ind * batch_size : (batch_ind + 1) * batch_size] += -loss

        dist.all_reduce(local_likelyhoods, op=dist.ReduceOp.SUM)

        cnt += 1
        k = len(imagenet_idx)
        truth_set = set(imagenet_idx)
        topk = local_likelyhoods.topk(k).indices.tolist()
        hits = sum(1 for c in topk if c in truth_set)
        running_hits += hits / k

        if rank == 0:
            print(filename, k, running_hits / cnt)

        if (
            checkpoint_dir is not None
            and (img_num + 1) % checkpoint_interval == 0
        ):
            dist.barrier()
            save_checkpoint(
                checkpoint_dir,
                rank,
                {
                    "img_num": img_num + 1,
                    "cnt": cnt,
                    "running_hits": running_hits,
                    "likelyhoods": likelyhoods.cpu(),
                    "g_state": g.get_state(),
                    "world_size": dist.get_world_size(),
                    "len_dataset": len_dataset,
                    "n_class": n_class,
                },
            )
            dist.barrier()

            if rank == 0:
                copy_out_to_snapshot('checkpoints')
            dist.barrier()

    if checkpoint_dir is not None and len(items) > skip_until:
        dist.barrier()
        save_checkpoint(
            checkpoint_dir,
            rank,
            {
                "img_num": len(items),
                "cnt": cnt,
                "running_hits": running_hits,
                "likelyhoods": likelyhoods.cpu(),
                "g_state": g.get_state(),
                "world_size": dist.get_world_size(),
                "len_dataset": len_dataset,
                "n_class": n_class,
            },
        )
        dist.barrier()

        if rank == 0:
            copy_out_to_snapshot('checkpoints')
        dist.barrier()

    dist.all_reduce(likelyhoods, op=dist.ReduceOp.SUM)
    return likelyhoods, truth_mask, ks


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--model", type=str, default="DiT-XL/2")
parser.add_argument("--downsample_size", type=int, default=8)

parser.add_argument("--dit_ckpt", type=str, default="model_weights/DiT_weights/DiT-XL-2-256x256.pt")

parser.add_argument("--coco_labels_path", type=str,
                    default="/home/iasudakov/COCO/val2017_imagenet_labels_filtered1000.json",
                    help="JSON file mapping {filename: {imagenet_idx: [...], ...}}")
parser.add_argument("--coco_images_dir", type=str,
                    default="/home/iasudakov/COCO/val2017",
                    help="Directory containing the COCO images.")

parser.add_argument("--step_size", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=125)
parser.add_argument("--use_augmentations", type=bool, default=False)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--restart", action="store_true",
                    help="Wipe existing checkpoint and start fresh.")

args = parser.parse_args()

if args.checkpoint_dir is None:
    args.checkpoint_dir = "checkpoints/dit_coco"

g = torch.Generator()
g.manual_seed(args.seed)


dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = rank
torch.cuda.set_device(device)
print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

dist.barrier()

############################## UPLOAD CHECKPOINT ########################

if rank == 0:
    copy_snapshot_to_out('checkpoints')
dist.barrier()

############################## LOAD COCO DATA ##############################

if rank == 0 and args.restart:
    if os.path.exists(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)
dist.barrier()

with open(args.coco_labels_path) as f:
    coco_labels = json.load(f)

items = [
    (fname, list(info["imagenet_idx"]))
    for fname, info in coco_labels.items()
    if len(info.get("imagenet_idx", [])) > 0
]

if rank == 0:
    print(
        f"Loaded {len(items)} COCO images with imagenet_idx labels "
        f"from {args.coco_labels_path}"
    )

# Always score against all 1000 ImageNet classes.
classes = torch.arange(1000).to(device)

############################## INIT MODELS ##############################
diffusion = create_diffusion(timestep_respacing="")

LOCAL_DIR = "model_weights/SiT_weights/sd-vae-ft-ema"
tokenizer = AutoencoderKL.from_pretrained(LOCAL_DIR, local_files_only=True).to(device)
tokenizer.eval()
dist.barrier()

model = DiT_models[args.model](
    input_size=args.image_size // args.downsample_size, num_classes=1000
).to(device)

ckpt = torch.load(args.dit_ckpt, map_location="cpu")
model.load_state_dict(ckpt)
model.eval()
model.to(torch.bfloat16)
dist.barrier()

############################## CALCULATE LIKELYHOODS ##############################

g = torch.Generator()
g.manual_seed(args.seed)
likelyhoods, truth_mask, ks = calculate_likelihoods(
    rank,
    tokenizer,
    model,
    diffusion,
    items,
    args.coco_images_dir,
    classes,
    args.batch_size,
    args.step_size,
    g,
    checkpoint_dir=args.checkpoint_dir,
    checkpoint_interval=args.checkpoint_interval,
)

############################## LOG METRICS ##############################

n_trials = 1000 // args.step_size

if rank == 0:
    likelyhoods_cpu = likelyhoods.cpu()
    truth_mask_cpu = truth_mask.cpu()
    ks_cpu = ks.cpu()

    n_scored = int((ks_cpu > 0).sum().item())
    print(f"COCO top-k recall over {n_scored} images (k = #imagenet labels per image)")

    for i in range(1, n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics_coco(
            likelyhoods_cpu, truth_mask_cpu, ks_cpu, i
        )
        print(f"acc_0_{i}_trials:", acc_i_0.mean(), "+-", acc_i_0.std())

    print("========================================")

    for i in range(1, n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics_coco(
            likelyhoods_cpu, truth_mask_cpu, ks_cpu, i
        )
        print(f"acc_1_{i}_trials:", acc_i_1.mean(), "+-", acc_i_1.std())

    print("========================================")

    for i in range(1, n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics_coco(
            likelyhoods_cpu, truth_mask_cpu, ks_cpu, i
        )
        print(f"acc_2_{i}_trials:", acc_i_2.mean(), "+-", acc_i_2.std())

    print("========================================")

dist.destroy_process_group()
