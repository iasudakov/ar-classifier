import argparse
import json
import os
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

from image_preprocessing import (
    center_crop_arr,
    random_crop_arr,
)


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
def calculate_likelihoods(rank, vae, model, diffusion, samples, classes, obj_to_im, batch_size, step_size, g):

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

    cnt = 0
    true = 0

    classes_list = classes.tolist() if torch.is_tensor(classes) else list(classes)

    for img_path, obj_id in samples:

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

args = parser.parse_args()

g = torch.Generator()
g.manual_seed(args.seed)


dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = rank
torch.cuda.set_device(device)
print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

dist.barrier()

############################## CREATE DATA ##############################

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
)

############################## LOG METRICS ##############################

n_trials = 1000 // args.step_size

if rank == 0:
    for i in range(1, n_trials + 1):
        acc_i = calc_statistics_objectnet(likelyhoods, targets, classes, obj_to_im, i)
        print(f"acc_{i}_trials:", acc_i)

dist.destroy_process_group()
