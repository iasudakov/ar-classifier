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
from image_preprocessing import (
    calc_statistics,
    center_crop_arr,
    random_crop_arr,
    sample_imagenet,
)


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
def _generate(model, latents, condition):
    condition = model.preprocess_condition(condition, cond_drop_prob=0.0)
    ids = latents[:, :-1]
    logits = forward_fn(model, ids, condition, orders=None, is_sampling=True)
    return logits


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
    valid=False,
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

    cnt = 0
    true = 0

    for class_ in classes:
        i = 0
        file_name = f"{data_path}/{class_}_{i}.JPEG"

        while os.path.exists(file_name):
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

                for batch_iter in range(batch_iters):
                    batch_ind = batch_iter + rank * batch_iters
                    ys = YS[batch_ind * batch_size : (batch_ind + 1) * batch_size]

                    logits = _generate(model, latents, condition=ys)
                    prob_seq = F.softmax(logits, dim=-1)
                    log_prob = prob_seq[
                        torch.arange(latents.shape[0])[:, None],
                        torch.arange(latents.shape[1])[None, :],
                        latents,
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
