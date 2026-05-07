import argparse
import json
import os
import shutil
import sys

import numpy as np
import PIL
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
sys.path.append("model_repositories/RandAR")
sys.path.append(".")
from RandAR.util import instantiate_from_config

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
    rank,
    tokenizer,
    model,
    items,
    images_dir,
    classes,
    batch_size,
    n_trials,
    g,
    use_raster,
    t=0,
    valid=False,
    checkpoint_dir=None,
    checkpoint_interval=1,
):

    if "maskgit" in config.tokenizer.target:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ]
        )

    device = model.cls_embedding.embedding_table.weight.device
    n_class = len(classes)
    len_dataset = len(items)

    YS = classes
    assert n_class % (batch_size * dist.get_world_size()) == 0
    batch_iters = n_class // (batch_size * dist.get_world_size())

    likelyhoods = torch.zeros((len_dataset, n_trials, n_class)).to(device)
    sum_logprobs = torch.zeros((len_dataset, n_class, args.image_size)).to(device)

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
                sum_logprobs.copy_(ckpt["sum_logprobs"].to(device))
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
        orig_img = PIL.Image.open(file_path).convert("RGB")

        likelyhoods_local = torch.zeros(n_class).to(device)

        for trial in range(n_trials):
            if args.use_augmentations:
                img = random_crop_arr(orig_img, args.image_size, g=g)
            else:
                img = center_crop_arr(orig_img, args.image_size)

            img = transform(img).to(device).unsqueeze(0)

            if "maskgit" in config.tokenizer.target:
                codes = tokenizer.encode_indices(img)
                latents = codes.reshape(img.shape[0], -1)[0]
                latents = latents.tile((batch_size, 1))
            else:
                h = tokenizer.encoder(img)
                h = tokenizer.quant_conv(h)
                h = (1 - t) * h + torch.randn_like(h) * t
                quant, emb_loss, info = tokenizer.quantize(h)
                latents = info[2]
                latents = latents.tile((batch_size, 1))

            token_order = torch.randperm(gpt_model.block_size, generator=g).to(
                device
            )
            if use_raster:
                token_order = torch.arange(gpt_model.block_size).to(device)
            if args.use_const_random:
                g_ = torch.Generator()
                g_.manual_seed(args.seed)
                token_order = torch.randperm(gpt_model.block_size, generator=g_).to(
                    device
                )

            token_order = token_order.unsqueeze(0).repeat(batch_size, 1)
            token_order = token_order.contiguous()

            for batch_iter in range(batch_iters):
                batch_ind = batch_iter + rank * batch_iters
                cond = YS[batch_ind * batch_size : (batch_ind + 1) * batch_size]
                with torch.autocast("cuda", torch.float):
                    logits, loss, token_order = gpt_model(latents, cond, targets=latents, token_order=token_order)
                    targets = (torch.gather(latents.unsqueeze(-1), 1, token_order.unsqueeze(-1)).squeeze(-1).contiguous())
                prob_seq = F.softmax(logits, dim=-1)

                log_prob = (prob_seq[torch.arange(latents.shape[0])[:, None], torch.arange(latents.shape[1])[None, :], targets]).log()
                likelyhoods[img_num, trial, batch_ind * batch_size : (batch_ind + 1) * batch_size] = log_prob.sum(dim=-1)
                sum_logprobs[img_num, batch_ind * batch_size : (batch_ind + 1) * batch_size] += log_prob
                likelyhoods_local[batch_ind * batch_size : (batch_ind + 1) * batch_size] += log_prob.sum(dim=-1)

        dist.all_reduce(likelyhoods_local, op=dist.ReduceOp.SUM)

        dist.barrier()

        cnt += 1
        k = len(imagenet_idx)
        truth_set = set(imagenet_idx)
        topk = likelyhoods_local.topk(k).indices.tolist()
        hits = sum(1 for c in topk if c in truth_set)
        running_hits += hits / k

        if rank == 0 and not valid:
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
                    "sum_logprobs": sum_logprobs.cpu(),
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
                "sum_logprobs": sum_logprobs.cpu(),
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
    dist.all_reduce(sum_logprobs, op=dist.ReduceOp.SUM)
    return likelyhoods, truth_mask, ks, sum_logprobs


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--config", type=str, default="model_repositories/RandAR/configs/randar/randar_l_0.3b_llamagen.yaml")
parser.add_argument("--downsample_size", type=int, default=16)


parser.add_argument("--coco_labels_path", type=str,
                    default="/home/iasudakov/COCO/val2017_imagenet_labels_filtered1000.json",
                    help="JSON file mapping {filename: {imagenet_idx: [...], ...}}")
parser.add_argument("--coco_images_dir", type=str,
                    default="/home/iasudakov/COCO/val2017",
                    help="Directory containing the COCO images.")
parser.add_argument("--gpt_ckpt", type=str, default="model_iters_00360000.pt")
parser.add_argument("--vq_ckpt", type=str, default="model_weights/RandAR_weights/vq_ds16_c2i.pt")
parser.add_argument("--n_trials", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=125)
parser.add_argument("--use_raster", type=bool, default=False)
parser.add_argument("--use_const_random", type=bool, default=False)
parser.add_argument("--use_augmentations", type=bool, default=False)
parser.add_argument("--t", type=float, default=0.0)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--restart", action="store_true",
                    help="Wipe existing checkpoint and start fresh.")


args = parser.parse_args()

if args.checkpoint_dir is None:
    args.checkpoint_dir = "checkpoints/randar_coco"

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
config = OmegaConf.load(args.config)
tokenizer = instantiate_from_config(config.tokenizer).to(device).eval()
ckpt = torch.load(args.vq_ckpt, map_location="cpu")
if "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt
tokenizer.load_state_dict(state_dict)
tokenizer.eval()

latent_size = args.image_size // args.downsample_size
gpt_model = instantiate_from_config(config.ar_model).to(
    device=device, dtype=torch.bfloat16
)

model_weight = torch.load(args.gpt_ckpt)
gpt_model.load_state_dict(model_weight, strict=True)
gpt_model.eval()
dist.barrier()


############################## CALCULATE LIKELIHOODS ##############################

g = torch.Generator()
g.manual_seed(args.seed)
likelyhoods, truth_mask, ks, sum_logprobs = calculate_likelihoods(
    rank,
    tokenizer,
    gpt_model,
    items,
    args.coco_images_dir,
    classes,
    args.batch_size,
    args.n_trials,
    g,
    args.use_raster,
    args.t,
    checkpoint_dir=args.checkpoint_dir,
    checkpoint_interval=args.checkpoint_interval,
)

############################## LOG METRICS ##############################

if rank == 0:
    likelyhoods_cpu = likelyhoods.cpu()
    truth_mask_cpu = truth_mask.cpu()
    ks_cpu = ks.cpu()

    n_scored = int((ks_cpu > 0).sum().item())
    print(f"COCO top-k recall over {n_scored} images (k = #imagenet labels per image)")

    for i in range(1, args.n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics_coco(
            likelyhoods_cpu, truth_mask_cpu, ks_cpu, i
        )
        print(f"acc_0_{i}_trials:", acc_i_0.mean(), "+-", acc_i_0.std())

    print("========================================")

    for i in range(1, args.n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics_coco(
            likelyhoods_cpu, truth_mask_cpu, ks_cpu, i
        )
        print(f"acc_1_{i}_trials:", acc_i_1.mean(), "+-", acc_i_1.std())

    print("========================================")

    for i in range(1, args.n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics_coco(
            likelyhoods_cpu, truth_mask_cpu, ks_cpu, i
        )
        print(f"acc_2_{i}_trials:", acc_i_2.mean(), "+-", acc_i_2.std())

    print("========================================")

dist.destroy_process_group()
