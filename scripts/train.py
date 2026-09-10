import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import SonarMaskDataset
from src.model import build_model


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * inter + 1e-6) / (den + 1e-6)).mean()


def calculate_loss(outputs, target):
    loss = (
        0.5 * F.binary_cross_entropy_with_logits(
            outputs["out"], target
        )
        + 0.5 * dice_loss(outputs["out"], target)
    )

    if outputs.get("aux") is not None:
        loss += 0.4 * F.binary_cross_entropy_with_logits(
            outputs["aux"], target
        )

    return loss


def validate(model, loader, device):
    model.eval()
    total = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            total += calculate_loss(outputs, y).item()

    return total / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepLabV3-ResNet50 for ghost-net segmentation"
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)

    print("Image size:", args.size)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)

    train_dataset = SonarMaskDataset(
        "data/ghostnet/train/images",
        "data/ghostnet/train/masks",
        args.size
    )

    val_dataset = SonarMaskDataset(
        "data/ghostnet/val/images",
        "data/ghostnet/val/masks",
        args.size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = build_model(True).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    Path("models").mkdir(exist_ok=True)

    checkpoint_path = Path("models/checkpoint.pt")
    best_path = Path("models/deeplabv3_resnet50_best.pt")

    start_epoch = 1
    best_val = float("inf")

    if args.resume and checkpoint_path.exists():
        print("Loading checkpoint...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint["best_val"]

        print(
            f"Resuming from epoch {start_epoch} "
            f"(best val loss: {best_val:.4f})"
        )

    for epoch in range(start_epoch, args.epochs + 1):

        model.train()
        train_total = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            loss = calculate_loss(outputs, y)

            loss.backward()
            optimizer.step()

            train_total += loss.item()

        train_avg = train_total / max(1, len(train_loader))
        val_avg = validate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train={train_avg:.4f} "
            f"val={val_avg:.4f}"
        )

        # Always save checkpoint after each completed epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": min(best_val, val_avg)
            },
            checkpoint_path
        )

        print("Checkpoint saved.")

        if val_avg < best_val:
            best_val = val_avg

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "size": args.size
                },
                best_path
            )

            print("Saved best model.")


if __name__ == "__main__":
    main()