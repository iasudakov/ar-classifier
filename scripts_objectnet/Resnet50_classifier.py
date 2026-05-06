import argparse
import json
import os
import time

import PIL
import torch
import torch.distributed as dist
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def all_reduce_sum(t):
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def barrier():
    if is_distributed():
        dist.barrier()


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


class ObjectNetDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, obj_id = self.samples[idx]
        img = PIL.Image.open(path).convert("RGB")
        return self.transform(img), obj_id


def topk_in_set(logits, obj_ids, obj_to_im, k):
    """For each row, return 1 if any of the top-k predictions is in obj_to_im[obj_id], else 0."""
    topk = logits.topk(k, dim=-1).indices.tolist()
    out = []
    for preds, obj_id in zip(topk, obj_ids):
        valid = obj_to_im[int(obj_id)]
        out.append(1 if any(p in valid for p in preds) else 0)
    return out


@torch.no_grad()
def evaluate(rank, world_size, model, loader, obj_to_im, device):
    candidate_ids = sorted({c for v in obj_to_im.values() for c in v})
    mask = torch.full((1000,), float("-inf"), device=device)
    mask[torch.tensor(candidate_ids, device=device)] = 0.0

    counters = torch.zeros(5, device=device)  # [top1_masked, top5_masked, top1_raw, top5_raw, total]
    n_total = len(loader.dataset)
    seen = 0
    t0 = time.time()

    for imgs, obj_ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        masked = logits + mask

        n = imgs.shape[0]
        counters[0] += sum(topk_in_set(masked, obj_ids, obj_to_im, 1))
        counters[1] += sum(topk_in_set(masked, obj_ids, obj_to_im, 5))
        counters[2] += sum(topk_in_set(logits, obj_ids, obj_to_im, 1))
        counters[3] += sum(topk_in_set(logits, obj_ids, obj_to_im, 5))
        counters[4] += n

        seen += n
        if rank == 0:
            elapsed = time.time() - t0
            done = int(counters[4].item())
            ips = done / elapsed if elapsed > 0 else 0.0
            print(
                f"[rank 0] {seen * world_size}/{n_total} "
                f"top1m={counters[0].item() / max(counters[4].item(), 1):.4f} "
                f"top5m={counters[1].item() / max(counters[4].item(), 1):.4f} "
                f"({ips:.1f} im/s/rank)",
                flush=True,
            )

    all_reduce_sum(counters)
    top1_m, top5_m, top1_r, top5_r, total = counters.tolist()
    return {
        "top1_masked": top1_m / total,
        "top5_masked": top5_m / total,
        "top1_raw": top1_r / total,
        "top5_raw": top5_r / total,
        "total": int(total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objectnet_path", type=str, default="objectnet_images/test")
    parser.add_argument("--objectnet_mapping", type=str, default="objectnet_images/objectnet_to_imagenet.json")
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V2", choices=["IMAGENET1K_V1", "IMAGENET1K_V2"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    rank, world_size, device = setup_distributed()
    if rank == 0:
        print(f"world_size={world_size} device={device}", flush=True)
    barrier()

    obj_to_im = load_objectnet_mapping(args.objectnet_mapping)
    samples = collect_objectnet_samples(args.objectnet_path, obj_to_im, max_per_class=args.max_per_class)
    if rank == 0:
        n_unique = len({c for v in obj_to_im.values() for c in v})
        print(
            f"ObjectNet: {len(obj_to_im)} folders, "
            f"{n_unique} unique ImageNet ids, {len(samples)} images.",
            flush=True,
        )
    barrier()

    weights = getattr(ResNet101_Weights, args.weights)
    transform = weights.transforms()
    model = resnet101(weights=weights).to(device).eval()
    barrier()

    dataset = ObjectNetDataset(samples, transform)
    if is_distributed():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
    else:
        sampler = None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    metrics = evaluate(rank, world_size, model, loader, obj_to_im, device)

    if rank == 0:
        print("=" * 60, flush=True)
        print(f"ResNet50 ({args.weights}) on ObjectNet ({metrics['total']} images)", flush=True)
        print(f"  top-1 masked (candidate classes only): {metrics['top1_masked']:.4f}", flush=True)
        print(f"  top-5 masked:                          {metrics['top5_masked']:.4f}", flush=True)
        print(f"  top-1 raw    (full 1000-way):          {metrics['top1_raw']:.4f}", flush=True)
        print(f"  top-5 raw:                             {metrics['top5_raw']:.4f}", flush=True)
        print("=" * 60, flush=True)

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
