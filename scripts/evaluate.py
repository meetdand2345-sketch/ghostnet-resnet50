import argparse
import torch
from torch.utils.data import DataLoader

from src.dataset import SonarMaskDataset
from src.model import build_model
from src.utils import load_threshold


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        default="models/deeplabv3_resnet50_best.pt"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=512
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    checkpoint = torch.load(
        args.weights,
        map_location=device
    )

    # Must match the model used during training.
    model = build_model(True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = SonarMaskDataset(
        "data/ghostnet/test/images",
        "data/ghostnet/test/masks",
        args.size
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    threshold = load_threshold()

    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():

        for x, y in loader:

            probabilities = torch.sigmoid(
                model(x.to(device))["out"]
            ).cpu()

            prediction = probabilities >= threshold
            ground_truth = y >= 0.5

            tp += (
                prediction & ground_truth
            ).sum().item()

            fp += (
                prediction & ~ground_truth
            ).sum().item()

            fn += (
                ~prediction & ground_truth
            ).sum().item()

    iou = tp / max(tp + fp + fn, 1)

    dice = (
        2 * tp /
        max(2 * tp + fp + fn, 1)
    )

    precision = tp / max(tp + fp, 1)

    recall = tp / max(tp + fn, 1)

    print(f"Threshold: {threshold:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
