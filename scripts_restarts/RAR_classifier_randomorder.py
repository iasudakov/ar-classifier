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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
sys.path.append("model_repositories/1d-tokenizer")
sys.path.append(".")

import demo_util
from utils.train_utils import create_pretrained_tokenizer

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
def forward_fn(model, input_ids, condition, orders=None, is_sampling=False):
    if orders is None:
        orders = model.get_raster_orders(input_ids)

    labels = input_ids.clone()
    input_ids = torch.cat(
        [condition.view(condition.shape[0], -1), input_ids.view(input_ids.shape[0], -1)], dim=1
    )
    embeddings = model.embeddings(input_ids)
    condition_token = embeddings[:, 0]

    pos_embed = model.pos_embed.repeat(input_ids.shape[0], 1, 1)
    prefix = 2
    pos_embed_prefix = pos_embed[:, :prefix]
    pos_embed_postfix = model.shuffle(pos_embed[:, prefix : prefix + model.image_seq_len], orders)

    target_aware_pos_embed = model.target_aware_pos_embed.repeat(input_ids.shape[0], 1, 1)
    target_aware_pos_embed_postfix = model.shuffle(
        target_aware_pos_embed[:, prefix : prefix + model.image_seq_len], orders
    )

    if not is_sampling:
        labels = model.shuffle(labels, orders)
        embeddings = torch.cat([embeddings[:, :1], model.shuffle(embeddings[:, 1:], orders)], dim=1)

    x = embeddings
    cls_tokens = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + torch.cat([pos_embed_prefix, pos_embed_postfix], dim=1)[:, : x.shape[1]]

    target_aware_pos_embed = torch.cat(
        [
            torch.zeros_like(x[:, : prefix - 1]),
            target_aware_pos_embed_postfix,
            torch.zeros_like(x[:, -1:]),
        ],
        dim=1,
    )
    x = x + target_aware_pos_embed[:, : x.shape[1]]

    attn_mask = model.attn_mask[: x.shape[1], : x.shape[1]]
    condition_token = condition_token.unsqueeze(1) + model.timesteps_embeddings[:, : x.shape[1]]

    if model.blocks[0].attn.kv_cache:
        if model.blocks[0].attn.k_cache is not None and model.blocks[0].attn.v_cache is not None:
            x = x[:, -1:]
            attn_mask = None
            condition_token = condition_token[:, -1:]

    for blk in model.blocks:
        x = blk(x, attn_mask=attn_mask, c=condition_token)

    if not model.blocks[0].attn.kv_cache:
        x = x[:, prefix - 1 :]
        condition_token = condition_token[:, prefix - 1 :]

    x = model.adaln_before_head(x, condition_token)
    x = model.lm_head(x)
    return x


@torch.no_grad()
def _generate(model, latents, condition, orders=None):
    condition = model.preprocess_condition(condition, cond_drop_prob=0.0)
    if orders is None:
        ids = latents[:, :-1]
        logits = forward_fn(model, ids, condition, orders=None, is_sampling=True)
        return logits, latents
    else:
        logits = forward_fn(model, latents, condition, orders=orders, is_sampling=False)
        logits = logits[:, :model.image_seq_len, :]
        shuffled_latents = model.shuffle(latents, orders)
        return logits, shuffled_latents


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
    use_raster=False,
    valid=False,
    checkpoint_dir=None,
    checkpoint_interval=1,
):
    transform = transforms.Compose([transforms.ToTensor()])

    device = next(model.parameters()).device
    n_class = len(classes)
    len_dataset = len(os.listdir(data_path))

    YS = classes
    assert n_class % (batch_size * dist.get_world_size()) == 0
    batch_iters = n_class // (batch_size * dist.get_world_size())

    likelyhoods = torch.zeros((len_dataset, n_trials, n_class)).to(device)
    sum_logprobs = torch.zeros((len_dataset, n_class, args.image_size)).to(device)
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
                sum_logprobs.copy_(ckpt["sum_logprobs"].to(device))
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
                if args.use_augmentations:
                    img = random_crop_arr(orig_img, args.image_size, g=g)
                else:
                    img = center_crop_arr(orig_img, args.image_size)

                img = transform(img).to(device).unsqueeze(0)

                with torch.no_grad():
                    latents = tokenizer.encode(img)
                    latents = latents[0].tile((batch_size, 1))

                if use_raster:
                    token_order = None
                elif args.use_const_random:
                    g_ = torch.Generator()
                    g_.manual_seed(args.seed)
                    token_order = torch.randperm(model.image_seq_len, generator=g_).to(device)
                    token_order = token_order.unsqueeze(0).repeat(batch_size, 1).contiguous()
                else:
                    token_order = torch.randperm(model.image_seq_len, generator=g).to(device)
                    token_order = token_order.unsqueeze(0).repeat(batch_size, 1).contiguous()

                for batch_iter in range(batch_iters):
                    batch_ind = batch_iter + rank * batch_iters
                    ys = YS[batch_ind * batch_size : (batch_ind + 1) * batch_size]

                    logits, targets = _generate(model, latents, condition=ys, orders=token_order)
                    prob_seq = F.softmax(logits, dim=-1)
                    log_prob = prob_seq[
                        torch.arange(latents.shape[0])[:, None],
                        torch.arange(latents.shape[1])[None, :],
                        targets,
                    ].log()
                    log_prob_sum = log_prob.sum(dim=-1)

                    likelyhoods[img_num, trial, batch_ind * batch_size : (batch_ind + 1) * batch_size] = log_prob_sum
                    sum_logprobs[img_num, batch_ind * batch_size : (batch_ind + 1) * batch_size] += log_prob
                    likelyhoods_local[batch_ind * batch_size : (batch_ind + 1) * batch_size] += log_prob_sum

            dist.all_reduce(likelyhoods_local, op=dist.ReduceOp.SUM)
            dist.barrier()

            cnt += 1
            if likelyhoods_local.argmax(dim=-1) == class_:
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
                        "sum_logprobs": sum_logprobs.cpu(),
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
                "sum_logprobs": sum_logprobs.cpu(),
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
    dist.all_reduce(sum_logprobs, op=dist.ReduceOp.SUM)
    return likelyhoods, targets_, sum_logprobs


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--config", type=str, default="model_repositories/1d-tokenizer/configs/training/generator/rar.yaml")
parser.add_argument("--downsample_size", type=int, default=16)

parser.add_argument("--dataset", type=str, default="val")
parser.add_argument("--imagenet_val_path", type=str)
parser.add_argument("--imagenet_X_path", type=str)
parser.add_argument("--weights_path", type=str, default="model_weights/RAR_weights")
parser.add_argument("--rar_model_size", type=str, default="rar_b")
parser.add_argument("--n_samples", type=int, default=2)
parser.add_argument("--n_trials", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=125)
parser.add_argument("--use_raster", action="store_true")
parser.add_argument("--use_const_random", action="store_true")
parser.add_argument("--use_augmentations", action="store_true")
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--restart", action="store_true",
                    help="Wipe existing checkpoint and dataset and start fresh.")

args = parser.parse_args()

if args.checkpoint_dir is None:
    args.checkpoint_dir = f"checkpoints/rar_randomorder_{args.dataset}"

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

config = demo_util.get_config(args.config)
config.experiment.generator_checkpoint = f"{args.weights_path}/{args.rar_model_size}.bin"
config.model.vq_model.pretrained_tokenizer_weight = f"{args.weights_path}/maskgit-vqgan-imagenet-f16-256.bin"
config.model.generator.hidden_size = {"rar_b": 768, "rar_l": 1024, "rar_xl": 1280, "rar_xxl": 1408}[args.rar_model_size]
config.model.generator.num_hidden_layers = {"rar_b": 24, "rar_l": 24, "rar_xl": 32, "rar_xxl": 40}[args.rar_model_size]
config.model.generator.num_attention_heads = 16
config.model.generator.intermediate_size = {"rar_b": 3072, "rar_l": 4096, "rar_xl": 5120, "rar_xxl": 6144}[args.rar_model_size]

tokenizer = create_pretrained_tokenizer(config)
generator = demo_util.get_rar_generator(config)
tokenizer.to(device)
generator.to(device)
tokenizer.eval()
generator.eval()
dist.barrier()


############################## CALCULATE LIKELIHOODS ##############################

g = torch.Generator()
g.manual_seed(args.seed)
likelyhoods, targets, sum_logprobs = calculate_likelihoods(
    rank,
    tokenizer,
    generator,
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
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics(likelyhoods, targets, i, classes)
        print(f"acc_0_{i}_trials:", acc_i_0.mean(), "+-", acc_i_0.std())

    print("========================================")

    for i in range(1, args.n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics(likelyhoods, targets, i, classes)
        print(f"acc_1_{i}_trials:", acc_i_1.mean(), "+-", acc_i_1.std())

    print("========================================")

    for i in range(1, args.n_trials + 1):
        acc_i_0, acc_i_1, acc_i_2 = calc_statistics(likelyhoods, targets, i, classes)
        print(f"acc_2_{i}_trials:", acc_i_2.mean(), "+-", acc_i_2.std())

    print("========================================")

dist.destroy_process_group()
