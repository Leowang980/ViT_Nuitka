import argparse
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import VisionTransformer, vit_base, vit_small, vit_tiny


MODEL_FACTORY = {
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vision Transformer training script.")
    parser.add_argument("--data-path", type=str, default="./data", help="Dataset root directory.")
    parser.add_argument("--model", type=str, default="vit_tiny", choices=MODEL_FACTORY.keys(), help="Model variant.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of target classes.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warm-up epochs for learning rate.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="torch device string or 'auto'.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save checkpoints.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Gradient clipping norm.")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(list(argv))


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(
    data_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, int]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    eval_transforms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transforms)
    val_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=eval_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    num_classes = len(getattr(train_dataset, "classes", [])) or 10
    return train_loader, val_loader, num_classes


def cosine_with_warmup_schedule(epochs: int, warmup_epochs: int) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0)


def train_one_epoch(
    model: VisionTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    epoch: int,
    max_grad_norm: Optional[float],
) -> Tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    start_time = time.time()
    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None and scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        acc = accuracy(outputs, targets)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

        if (step + 1) % 50 == 0 or (step + 1) == len(dataloader):
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch+1}] Step [{step+1}/{len(dataloader)}] "
                f"Loss: {loss_meter.avg:.4f} Acc: {acc_meter.avg*100:.2f}% "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} "
                f"Time: {elapsed:.1f}s",
                flush=True,
            )
            start_time = time.time()

    return loss_meter.avg, acc_meter.avg


def validate(
    model: VisionTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
) -> Tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(accuracy(outputs, targets), images.size(0))

    return loss_meter.avg, acc_meter.avg


def save_checkpoint(
    output_dir: Path,
    state: Dict,
    is_best: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / "last.pth")
    if is_best:
        torch.save(state, output_dir / "best.pth")


def load_checkpoint(
    path: Path,
    model: VisionTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_acc = checkpoint.get("best_acc", 0.0)
    print(f"Resumed from checkpoint '{path}' (epoch {start_epoch})")
    return start_epoch, best_acc


def run_training(args: argparse.Namespace) -> float:
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    effective_workers = args.num_workers
    if device.type == "mps" and effective_workers > 0:
        print("MPS detected: forcing DataLoader num_workers=0 for compatibility.")
        effective_workers = 0

    pin_memory = device.type == "cuda"

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    train_loader, val_loader, inferred_classes = build_dataloaders(
        data_path=data_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=effective_workers,
        pin_memory=pin_memory,
    )
    num_classes = args.num_classes or inferred_classes

    if args.model not in MODEL_FACTORY:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(MODEL_FACTORY.keys())}")
    model = MODEL_FACTORY[args.model](num_classes=num_classes, image_size=args.image_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=cosine_with_warmup_schedule(args.epochs, args.warmup_epochs))
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if device.type == "cuda" and args.amp:
        scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            start_epoch, best_acc = load_checkpoint(resume_path, model, optimizer, scheduler, scaler)
        else:
            print(f"Checkpoint '{resume_path}' not found. Starting from scratch.")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            args.max_grad_norm,
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device, amp=args.amp and device.type == "cuda")
        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc*100:.2f}% "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc*100:.2f}%"
        )

        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_acc": best_acc,
            "args": vars(args),
        }
        save_checkpoint(output_dir, state, is_best)

    print(f"Training finished. Best Val Acc: {best_acc*100:.2f}%")
    return best_acc


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
