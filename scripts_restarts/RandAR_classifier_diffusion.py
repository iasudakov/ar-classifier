import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
sys.path.append("model_repositories/RandAR_DM")
sys.path.append(".")
from RandAR.util import instantiate_from_config

import sys
sys.path.append('/home/iasudakov/')

from yt_tools.nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot

from image_preprocessing import (
    calc_statistics,
    center_crop_arr,
    random_crop_arr,
    sample_imagenet,
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


@torch.no_grad()
def calculate_likelihoods(
    rank,
    tokenizer,
    model,
    data_path,
    classes,
    batch_size,
    n_trials,
    g,
    use_raster,
    valid=False,
    checkpoint_dir=None,
    checkpoint_interval=1,
):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ]
    )

    device = model.cls_embedding.embedding_table.weight.device
    n_class = len(classes)
    len_dataset = len(os.listdir(data_path))

    YS = classes
    assert n_class % (batch_size * dist.get_world_size()) == 0
    batch_iters = n_class // (batch_size * dist.get_world_size())

    # DDM mask schedule: midpoint-bucket counts of "unmasked" tokens, one per trial.
    block_size = model.block_size
    step_size = block_size / n_trials
    rs = torch.arange(step_size / 2, block_size, step_size).to(torch.int32)
    assert len(rs) == n_trials, f"Expected {n_trials} mask levels, got {len(rs)}"
    if rank == 0:
        print(f"DDM unmasked-counts schedule: {rs.tolist()}")

    likelyhoods = torch.zeros((len_dataset, n_trials, n_class)).to(device)

    targets_ = torch.zeros(len_dataset).long().to(device)
    img_num = 0
    skip_until = 0

    cnt = 0
    true = 0

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

    for class_ in classes:
        i = 0
        file_name = f"{data_path}/{class_}_{i}.JPEG"

        while os.path.exists(file_name):
            if img_num < skip_until:
                i += 1
                file_name = f"{data_path}/{class_}_{i}.JPEG"
                img_num += 1
                continue

            orig_img = PIL.Image.open(file_name).convert("RGB")

            likelyhoods_local = torch.zeros(n_class).to(device)

            for trial in range(n_trials):
                unmasked = int(rs[trial].item())

                if args.use_augmentations:
                    img = random_crop_arr(orig_img, args.image_size, g=g)
                else:
                    img = center_crop_arr(orig_img, args.image_size)

                img = transform(img).to(device).unsqueeze(0)

                h = tokenizer.encoder(img)
                h = tokenizer.quant_conv(h)
                quant, emb_loss, info, _ = tokenizer.quantize(h)
                latents = info[2]
                latents = latents.tile((batch_size, 1))

                token_order = torch.randperm(block_size, generator=g).to(device)
                if use_raster:
                    token_order = torch.arange(block_size).to(device)
                if args.use_const_random:
                    g_ = torch.Generator()
                    g_.manual_seed(args.seed)
                    token_order = torch.randperm(block_size, generator=g_).to(device)

                token_order = token_order.unsqueeze(0).repeat(batch_size, 1)
                token_order = token_order.contiguous()

                for batch_iter in range(batch_iters):
                    batch_ind = batch_iter + rank * batch_iters
                    cond = YS[batch_ind * batch_size : (batch_ind + 1) * batch_size]
                    with torch.autocast("cuda", torch.float):
                        logits = gpt_model(
                            unmasked, latents, cond, targets=latents, token_order=token_order
                        )[:, -block_size + unmasked:]
                        targets = torch.gather(
                            latents.unsqueeze(-1), 1, token_order.unsqueeze(-1)
                        ).squeeze(-1).contiguous()[:, unmasked:]
                    prob_seq = F.softmax(logits, dim=-1)

                    log_prob = (
                        prob_seq[
                            torch.arange(latents.shape[0])[:, None],
                            torch.arange(targets.shape[1])[None, :],
                            targets,
                        ]
                    ).log()

                    likelyhoods[img_num, trial, batch_ind * batch_size : (batch_ind + 1) * batch_size] = log_prob.mean(dim=-1)
                    likelyhoods_local[batch_ind * batch_size : (batch_ind + 1) * batch_size] += log_prob.mean(dim=-1)

            dist.all_reduce(likelyhoods_local, op=dist.ReduceOp.SUM)

            dist.barrier()

            cnt += 1
            if YS[likelyhoods_local.argmax(dim=-1)] == class_:
                true += 1

            if rank == 0 and not valid:
                print(class_, true / cnt)

            i += 1
            file_name = f"{data_path}/{class_}_{i}.JPEG"

            targets_[img_num] = class_
            img_num += 1

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
parser.add_argument("--config", type=str, default="model_repositories/RandAR/configs/randar/randar_l_0.3b_llamagen.yaml")
parser.add_argument("--downsample_size", type=int, default=16)


parser.add_argument("--dataset", type=str, default="val")
parser.add_argument("--imagenet_val_path", type=str)
parser.add_argument("--imagenet_X_path", type=str)
parser.add_argument("--gpt_ckpt", type=str, default="model_iters_00360000.pt")
parser.add_argument("--vq_ckpt", type=str, default="model_weights/RandAR_weights/vq_ds16_c2i.pt")
parser.add_argument("--n_samples", type=int, default=2)
parser.add_argument("--n_trials", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=125)
parser.add_argument("--use_raster", type=bool, default=False)
parser.add_argument("--use_const_random", type=bool, default=False)
parser.add_argument("--use_augmentations", type=bool, default=False)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--restart", action="store_true",
                    help="Wipe existing checkpoint and dataset and start fresh.")


args = parser.parse_args()

if args.checkpoint_dir is None:
    args.checkpoint_dir = f"checkpoints/randar_{args.dataset}"

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

data_path = f"imagenet_data/imagenet_{args.dataset}"
classes_npy = f"{data_path}.npy"

if rank == 0 and args.restart:
    if os.path.exists(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    if os.path.exists(classes_npy):
        os.remove(classes_npy)
dist.barrier()

data_exists = (
    os.path.exists(classes_npy)
    and os.path.isdir(data_path)
    and len(os.listdir(data_path)) > 0
)

if rank == 0:
    if not data_exists:
        folder = Path(data_path)
        if folder.exists():
            shutil.rmtree(folder)

        classes = sample_imagenet(
            args.imagenet_X_path,
            args.imagenet_val_path,
            data_path,
            N_SAMPLES=args.n_samples,
        )
        np.save(f"{data_path}", classes)
        print(f"Created new dataset at {data_path}")
    else:
        print(f"Reusing existing dataset at {data_path}")
dist.barrier()

classes = torch.tensor(np.load(classes_npy)).to(device)


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
likelyhoods, targets = calculate_likelihoods(
    rank,
    tokenizer,
    gpt_model,
    data_path,
    classes,
    args.batch_size,
    args.n_trials,
    g,
    args.use_raster,
    checkpoint_dir=args.checkpoint_dir,
    checkpoint_interval=args.checkpoint_interval,
)

############################## LOG METRICS ##############################

if rank == 0:
    for i in range(1, args.n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics(likelyhoods, targets, i)
        print(f"acc_0_{i}_trials:", acc_i_0.mean(), "+-", acc_i_0.std())

    print("========================================")

    for i in range(1, args.n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics(likelyhoods, targets, i)
        print(f"acc_1_{i}_trials:", acc_i_1.mean(), "+-", acc_i_1.std())

    print("========================================")

    for i in range(1, args.n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics(likelyhoods, targets, i)
        print(f"acc_2_{i}_trials:", acc_i_2.mean(), "+-", acc_i_2.std())

    print("========================================")

dist.destroy_process_group()
