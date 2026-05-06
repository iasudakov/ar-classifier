import argparse
import math
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
sys.path.append("model_repositories/Maskgit-pytorch")
sys.path.append(".")

from Network.transformer import MaskTransformer
from Network.Taming.models.vqgan import VQModel

from image_preprocessing import (
    calc_statistics,
    center_crop_arr,
    sample_imagenet,
)


def get_masking_ratios(n_masks, mode="arccos"):
    """Return n_masks evenly-spaced masking ratios using the given schedule.

    Analogous to DiT's fixed timestep grid: ts = arange(step//2, 1000, step).
    Points are in the open interval (0, 1) to avoid degenerate all-visible /
    all-masked cases.
    """
    # n_masks evenly-spaced points in (0, 1) exclusive
    r = torch.linspace(0, 1, n_masks + 2)[1:-1]
    if mode == "linear":
        ratios = r
    elif mode == "square":
        ratios = r ** 2
    elif mode == "cosine":
        ratios = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":
        # arccos maps (0,1) → (1,0) fraction-to-mask, matching MaskGIT training
        ratios = torch.arccos(r) / (math.pi * 0.5)
    else:
        raise ValueError(f"Unknown mask schedule mode: {mode}")
    return ratios.tolist()


@torch.no_grad()
def calculate_likelihoods(
    rank, vqgan, vit, data_path, classes, batch_size,
    n_masks, mask_mode, codebook_size, patch_size, g,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    mask_token = codebook_size  # special mask token index
    device = next(vit.parameters()).device
    n_class = len(classes)
    n_tokens = patch_size * patch_size

    len_dataset = len(os.listdir(data_path))

    YS = classes
    assert n_class % (batch_size * dist.get_world_size()) == 0
    batch_iters = n_class // (batch_size * dist.get_world_size())

    # Fixed masking ratios (one random mask sampled per ratio)
    masking_ratios = get_masking_ratios(n_masks, mode=mask_mode)

    likelyhoods = torch.zeros((len_dataset, n_masks, n_class)).to(device)
    targets_ = torch.zeros(len_dataset).long().to(device)
    img_num = 0
    cnt = 0
    true = 0

    # Never drop class label during evaluation
    drop_label = torch.zeros(batch_size, dtype=torch.bool).to(device)

    for class_ in classes:
        i = 0
        file_name = f"{data_path}/{class_}_{i}.JPEG"

        while os.path.exists(file_name):
            local_likelyhoods = torch.zeros(n_class).to(device)

            orig_img = PIL.Image.open(file_name).convert("RGB")

            img = center_crop_arr(orig_img, args.image_size)
            img = transform(img).to(device).unsqueeze(0)
            _, _, [_, _, code] = vqgan.encode(img)
            code = code.reshape(1, patch_size, patch_size)  # (1, H, W)

            for trial, ratio in enumerate(masking_ratios):

                # Sample a random binary mask at this masking ratio
                mask_flat = (torch.rand(n_tokens, generator=g) < ratio).to(device)  # (n_tokens,)
                n_masked = int(mask_flat.sum().item())
                if n_masked == 0:
                    continue

                # Apply mask: replace masked positions with mask token
                masked_code = code.clone()
                masked_code.view(-1)[mask_flat] = mask_token

                # Tile masked code across all classes in the batch
                masked_code_tiled = masked_code.tile(batch_size, 1, 1)  # (B, H, W)

                # Ground-truth tokens at masked positions (same for all classes)
                code_flat = code.view(-1)                       # (n_tokens,)
                masked_targets = code_flat[mask_flat]           # (n_masked,)
                masked_targets_tiled = (masked_targets.unsqueeze(0).expand(batch_size, -1).reshape(-1))

                for batch_iter in range(batch_iters):
                    batch_ind = batch_iter + rank * batch_iters
                    cond = YS[batch_ind * batch_size : (batch_ind + 1) * batch_size]

                    with torch.autocast(device.type, torch.bfloat16):
                        logits = vit(masked_code_tiled, cond, drop_label=drop_label)
                        # logits: (B, n_tokens, codebook_size+1)

                    # Extract logits at masked positions and cast to float for CE
                    # masked_logits: (B, n_masked, codebook_size+1)
                    masked_logits = logits[:, mask_flat, :].float()
                    masked_logits_flat = masked_logits.reshape(-1, codebook_size + 1)

                    # Cross-entropy averaged over masked tokens → ELBO estimate
                    # Analogous to DiT averaging MSE over all noise dimensions
                    ce_per_token = F.cross_entropy(masked_logits_flat, masked_targets_tiled, reduction='none')
                    loss_per_class = ce_per_token.reshape(batch_size, n_masked).mean(dim=1)

                    likelyhoods[img_num, trial, batch_ind*batch_size : (batch_ind + 1)*batch_size] = -loss_per_class
                    local_likelyhoods[batch_ind*batch_size : (batch_ind + 1)*batch_size] += -loss_per_class

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

    dist.all_reduce(likelyhoods, op=dist.ReduceOp.SUM)
    return likelyhoods, targets_


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--image_size", type=int, default=256)

parser.add_argument("--vqgan_config", type=str,
                    default="model_repositories/Maskgit-pytorch/pretrained_maskgit/VQGAN/model.yaml")
parser.add_argument("--vqgan_ckpt", type=str,
                    default="model_repositories/Maskgit-pytorch/pretrained_maskgit/VQGAN/last.ckpt")
parser.add_argument("--vit_ckpt", type=str,
                    default="model_repositories/Maskgit-pytorch/pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth")

parser.add_argument("--dataset", type=str, default="val")
parser.add_argument("--imagenet_val_path", type=str)
parser.add_argument("--imagenet_X_path", type=str)

parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--n_masks", type=int, default=100,
                    help="Number of masking ratios to average over (analogous to n_timesteps in DiT)")
parser.add_argument("--mask_mode", type=str, default="arccos",
                    choices=["arccos", "linear", "cosine", "square"],
                    help="Schedule for masking ratios (arccos matches MaskGIT training)")
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

data_path = f"imagenet_data/imagenet_{args.dataset}"
if rank == 0:
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
dist.barrier()

classes = torch.tensor(np.load(f"{data_path}.npy")).to(device)

############################## INIT MODELS ##############################

# Load VQGAN
vqgan_config = OmegaConf.load(args.vqgan_config)
vqgan = VQModel(**vqgan_config.model.params)
vqgan_ckpt = torch.load(args.vqgan_ckpt, map_location="cpu")["state_dict"]
vqgan.load_state_dict(vqgan_ckpt, strict=False)
vqgan = vqgan.eval().to(device)

codebook_size = vqgan.n_embed  # 1024 for the standard VQGAN
patch_size = args.image_size // 2 ** (vqgan.encoder.num_resolutions - 1)
print(f"Codebook size: {codebook_size}, patch size: {patch_size}x{patch_size}")
dist.barrier()

# Load MaskTransformer
vit = MaskTransformer(
    img_size=args.image_size,
    hidden_dim=768,
    codebook_size=codebook_size,
    depth=24,
    heads=16,
    mlp_dim=3072,
    dropout=0.1,
)
vit_ckpt = torch.load(args.vit_ckpt, map_location="cpu")
vit.load_state_dict(vit_ckpt["model_state_dict"], strict=False)
vit = vit.eval().to(device)
dist.barrier()


total_params = sum(p.numel() for p in vit.parameters())
print(f"Total parameters: {total_params}")

############################## CALCULATE LIKELYHOODS ##############################

g = torch.Generator()
g.manual_seed(args.seed)
likelyhoods, targets = calculate_likelihoods(
    rank,
    vqgan,
    vit,
    data_path,
    classes,
    args.batch_size,
    args.n_masks,
    args.mask_mode,
    codebook_size,
    patch_size,
    g,
)

############################## LOG METRICS ##############################

if rank == 0:
    for i in range(1, args.n_masks + 1):
        acc_i = calc_statistics(likelyhoods, targets, i, classes)
        print(f"acc_{i}_masks:", acc_i)

dist.destroy_process_group()
