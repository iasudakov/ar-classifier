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


def load_objectnet_mapping(path):
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): list(v) for k, v in raw.items()}


def collect_objectnet_samples(data_path, obj_to_im, max_per_class=None):
    samples = []
    for obj_id_str in sorted(os.listdir(data_path), key=lambda x: int(x) if x.isdigit() else x):
        cls_dir = os.path.join(data_path, obj_id_str)
        if not os.path.isdir(cls_dir):
            continue
        try:
            obj_id = int(obj_id_str)
        except ValueError:
            continue
        if obj_id not in obj_to_im:
            continue
        files = sorted(os.listdir(cls_dir))
        if max_per_class is not None:
            files = files[:max_per_class]
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(cls_dir, fname), obj_id))
    return samples


def build_candidate_classes(obj_to_im, pad_unit):
    unique_ids = sorted({c for v in obj_to_im.values() for c in v})
    n = len(unique_ids)
    if n % pad_unit != 0:
        pad_to = ((n // pad_unit) + 1) * pad_unit
        unique_ids = unique_ids + [unique_ids[0]] * (pad_to - n)
    return unique_ids


def calc_statistics_objectnet(likelyhoods, targets, classes, obj_to_im, n_trials_acc):
    len_dataset, n_trials, n_class = likelyhoods.shape
    n_averaging = n_trials // n_trials_acc

    classes_list = classes.tolist() if torch.is_tensor(classes) else list(classes)
    allowed = torch.zeros(len_dataset, n_class, dtype=torch.bool, device=likelyhoods.device)
    for i in range(len_dataset):
        obj_id = int(targets[i].item())
        valid_set = set(obj_to_im[obj_id])
        for j, c in enumerate(classes_list):
            if int(c) in valid_set:
                allowed[i, j] = True

    idx = torch.arange(len_dataset, device=likelyhoods.device)
    accuracy_list_0, accuracy_list_1, accuracy_list_2 = [], [], []
    for i in range(n_averaging):
        chunk = likelyhoods[:, i * n_trials_acc : (i + 1) * n_trials_acc]

        pred_0 = chunk.mean(dim=1).argmax(dim=-1)
        accuracy_list_0.append(allowed[idx, pred_0].float().mean().item())

        pred_1 = torch.softmax(chunk, dim=-1).mean(dim=1).argmax(dim=-1)
        accuracy_list_1.append(allowed[idx, pred_1].float().mean().item())

        pred_2 = torch.logsumexp(chunk, dim=1).argmax(dim=-1)
        accuracy_list_2.append(allowed[idx, pred_2].float().mean().item())

    return (
        np.array(accuracy_list_0),
        np.array(accuracy_list_1),
        np.array(accuracy_list_2),
    )


@torch.no_grad()
def calculate_likelihoods(
    rank, vae, model, diffusion, samples, classes, obj_to_im, batch_size, step_size, g,
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
    len_dataset = len(samples)

    YS = classes
    assert n_class % (batch_size * dist.get_world_size()) == 0
    batch_iters = n_class // (batch_size * dist.get_world_size())

    ts = np.arange(step_size // 2, 1000, step_size)

    likelyhoods = torch.zeros((len_dataset, n_trials, n_class)).to(device)
    targets_ = torch.zeros(len_dataset).long().to(device)
    img_num = 0
    skip_until = 0

    cnt = 0
    true = 0

    classes_list = classes.tolist() if torch.is_tensor(classes) else list(classes)

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
                targets_.copy_(ckpt["targets_"].to(device))
                skip_until = int(ckpt["img_num"])
                cnt = int(ckpt.get("cnt", 0))
                true = int(ckpt.get("true", 0))
                g.set_state(ckpt["g_state"])
                if rank == 0:
                    print(
                        f"Resuming from checkpoint at img_num={skip_until} "
                        f"(cnt={cnt}, true={true})"
                    )
        dist.barrier()

    for img_path, obj_id in samples:
        if img_num < skip_until:
            img_num += 1
            continue

        local_likelyhoods = torch.zeros(n_class).to(device)

        for trial, t in enumerate(ts):

            orig_img = PIL.Image.open(img_path).convert("RGB")

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
        pred_idx = local_likelyhoods.argmax().item()
        pred_imagenet = int(classes_list[pred_idx])
        if pred_imagenet in obj_to_im[obj_id]:
            true += 1

        targets_[img_num] = obj_id
        img_num += 1

        if rank == 0:
            print(img_num, obj_id, pred_imagenet, true / cnt)

        if (
            checkpoint_dir is not None
            and img_num % checkpoint_interval == 0
        ):
            dist.barrier()
            save_checkpoint(
                checkpoint_dir,
                rank,
                {
                    "img_num": img_num,
                    "cnt": cnt,
                    "true": true,
                    "likelyhoods": likelyhoods.cpu(),
                    "targets_": targets_.cpu(),
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

    if checkpoint_dir is not None and img_num > skip_until:
        dist.barrier()
        save_checkpoint(
            checkpoint_dir,
            rank,
            {
                "img_num": img_num,
                "cnt": cnt,
                "true": true,
                "likelyhoods": likelyhoods.cpu(),
                "targets_": targets_.cpu(),
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
    return likelyhoods, targets_


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--model", type=str, default="DiT-XL/2")
parser.add_argument("--downsample_size", type=int, default=8)

parser.add_argument("--dit_ckpt", type=str, default="model_weights/DiT_weights/DiT-XL-2-256x256.pt")

parser.add_argument("--objectnet_path", type=str, default="objectnet_images/test")
parser.add_argument("--objectnet_mapping", type=str, default="objectnet_images/objectnet_to_imagenet.json")
parser.add_argument("--max_per_class", type=int, default=None)

parser.add_argument("--step_size", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=125)
parser.add_argument("--use_augmentations", type=bool, default=False)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--restart", action="store_true",
                    help="Wipe existing checkpoint and start fresh.")

args = parser.parse_args()

if args.checkpoint_dir is None:
    args.checkpoint_dir = "checkpoints/dit_objectnet"

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

############################## CREATE DATA ##############################

if rank == 0 and args.restart:
    if os.path.exists(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)
dist.barrier()

obj_to_im = load_objectnet_mapping(args.objectnet_mapping)
candidate_ids = build_candidate_classes(obj_to_im, args.batch_size * world_size)
classes = torch.tensor(candidate_ids).to(device)

samples = collect_objectnet_samples(
    args.objectnet_path, obj_to_im, max_per_class=args.max_per_class
)
if rank == 0:
    print(
        f"ObjectNet: {len(obj_to_im)} folders, "
        f"{len(set(candidate_ids))} unique ImageNet ids "
        f"(padded to {len(candidate_ids)}), {len(samples)} images."
    )
dist.barrier()

############################## INIT MODELS ##############################
diffusion = create_diffusion(timestep_respacing="")

tokenizer = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-ema",
    cache_dir="model_weights/DiT_weights",
).to(device)
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
likelyhoods, targets = calculate_likelihoods(
    rank,
    tokenizer,
    model,
    diffusion,
    samples,
    classes,
    obj_to_im,
    args.batch_size,
    args.step_size,
    g,
    checkpoint_dir=args.checkpoint_dir,
    checkpoint_interval=args.checkpoint_interval,
)

############################## LOG METRICS ##############################

n_trials = 1000 // args.step_size

if rank == 0:
    for i in range(1, n_trials + 1):
        acc_i = calc_statistics_objectnet(likelyhoods, targets, classes, obj_to_im, i)
        print(f"acc_{i}_trials:", acc_i)

dist.destroy_process_group()
