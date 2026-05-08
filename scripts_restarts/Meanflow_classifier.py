import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from diffusers.models import AutoencoderKL

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
sys.path.append("model_repositories/MeanFlow")
sys.path.append(".")

from sit import SiT_models, FinalLayer

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
    rank, vae, model, data_path, classes, batch_size, step_size, g,
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

    len_dataset = len(os.listdir(data_path))

    YS = classes
    assert n_class % (batch_size * dist.get_world_size()) == 0
    batch_iters = n_class // (batch_size * dist.get_world_size())

    # Flow matching uses t in [0, 1]
    ts = (np.arange(step_size // 2, 1000, step_size) / 1000).tolist()

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

            local_likelyhoods = torch.zeros(n_class).to(device)

            orig_img = PIL.Image.open(file_name).convert("RGB")

            # Encode once per image when not using augmentations
            if not args.use_augmentations:
                img = center_crop_arr(orig_img, args.image_size)
                img = transform(img).to(device).unsqueeze(0)
                latents = vae.encode(img).latent_dist.sample().mul_(0.18215)

            for trial, t_val in enumerate(ts):

                if args.use_augmentations:
                    img = random_crop_arr(orig_img, args.image_size, g=g)
                    img = transform(img).to(device).unsqueeze(0)
                    latents = vae.encode(img).latent_dist.sample().mul_(0.18215)

                t = torch.tensor(t_val, device=device)
                noise = torch.randn(latents.shape, generator=g).to(device)

                # Flow matching interpolation: x_t = (1-t)*x_0 + t*eps
                noised_latents = (1 - t) * latents + t * noise
                noised_latents = noised_latents.tile((batch_size, 1, 1, 1))

                t_batch = torch.full((batch_size,), t_val, device=device)
                r_batch = torch.full((batch_size,), 0.0, device=device)
                t_w = t.view(1, 1, 1, 1)  # broadcast shape for spatial dims

                for batch_iter in range(batch_iters):
                    batch_ind = batch_iter + rank * batch_iters
                    cond = YS[batch_ind * batch_size : (batch_ind + 1) * batch_size]

                    with torch.autocast(device.type, torch.bfloat16):
                        u = model(noised_latents, r_batch, t_batch, y=cond)

                    x_0_pred = noised_latents - t_w * u
                    eps_pred = (noised_latents - (1 - t_w) * x_0_pred) / t_w
                    loss = ((eps_pred - noise) ** 2).mean(dim=(1, 2, 3))

                    likelyhoods[img_num, trial, batch_ind * batch_size : (batch_ind + 1) * batch_size] = -loss
                    local_likelyhoods[batch_ind * batch_size : (batch_ind + 1) * batch_size] += -loss

            dist.all_reduce(local_likelyhoods, op=dist.ReduceOp.SUM)

            cnt += 1
            if YS[local_likelyhoods.argmax()] == class_:
                true += 1

            i += 1
            file_name = f"{data_path}/{class_}_{i}.JPEG"

            targets_[img_num] = class_
            img_num += 1

            if rank == 0:
                print(img_num, true / cnt)

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
parser.add_argument("--model", type=str, default="SiT-L/2")
parser.add_argument("--downsample_size", type=int, default=8)

parser.add_argument("--ckpt", type=str, default="model_weights/SiT_weights/sit_l_2_meanflow_ema.pt")

parser.add_argument("--dataset", type=str, default="val")
parser.add_argument("--imagenet_val_path", type=str)
parser.add_argument("--imagenet_X_path", type=str)

parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--step_size", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--use_augmentations", type=bool, default=False)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--restart", action="store_true",
                    help="Wipe existing checkpoint and dataset and start fresh.")

args = parser.parse_args()

if args.checkpoint_dir is None:
    args.checkpoint_dir = f"checkpoints/meanflow_{args.dataset}"

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
LOCAL_DIR = "model_weights/SiT_weights/sd-vae-ft-ema"
vae = AutoencoderKL.from_pretrained(LOCAL_DIR, local_files_only=True).to(device)
vae.eval()
dist.barrier()

block_kwargs = {"fused_attn": False, "qk_norm": False}
latent_size = args.image_size // args.downsample_size
model = SiT_models[args.model](
    input_size=latent_size,
    num_classes=1000,
    use_cfg=True,
    **block_kwargs,
).to(device)

state_dict = torch.load(args.ckpt, map_location="cpu")

# Released MeanFlow checkpoints disagree on final-layer width:
# SiT-XL/2 keeps the original-SiT learn_sigma layout (out_channels = 2*in_channels)
# while SiT-L/2 uses out_channels = in_channels. Match whichever the checkpoint has.
ckpt_out_dim = state_dict["final_layer.linear.weight"].shape[0]
ckpt_out_channels = ckpt_out_dim // (model.patch_size ** 2)
if ckpt_out_channels != model.out_channels:
    model.out_channels = ckpt_out_channels
    decoder_hidden_size = model.final_layer.adaLN_modulation[-1].out_features // 2
    model.final_layer = FinalLayer(decoder_hidden_size, model.patch_size, model.out_channels).to(device)
    if ckpt_out_channels == 2 * model.in_channels:
        _orig_forward = model.forward
        def _forward_chunk(*a, **kw):
            out = _orig_forward(*a, **kw)
            v, _ = out.chunk(2, dim=1)
            return v
        model.forward = _forward_chunk

model.load_state_dict(state_dict)
model.eval()
model.to(torch.bfloat16)
dist.barrier()

############################## CALCULATE LIKELYHOODS ##############################

g = torch.Generator()
g.manual_seed(args.seed)
likelyhoods, targets = calculate_likelihoods(
    rank,
    vae,
    model,
    data_path,
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
    for i in range(1, n_trials + 1):
        acc_i = calc_statistics(likelyhoods, targets, i, classes)
        print(f"acc_{i}_trials:", acc_i)

dist.destroy_process_group()
