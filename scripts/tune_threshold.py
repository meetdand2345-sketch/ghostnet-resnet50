import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import SonarMaskDataset
from src.model import build_model
from src.utils import save_threshold


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
        "data/ghostnet/val/images",
        "data/ghostnet/val/masks",
        args.size
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    cache = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            probabilities = torch.sigmoid(
                model(x)["out"]
            ).cpu()

            cache.append((probabilities, y))

    best_iou = -1.0
    best_threshold = 0.5

    for threshold in np.arange(0.10, 0.91, 0.05):

        scores = []

        for probabilities, masks in cache:

            for probability, mask in zip(
                probabilities,
                masks
            ):

                prediction = probability >= threshold
                ground_truth = mask >= 0.5

                intersection = (
                    prediction & ground_truth
                ).sum().item()

                union = (
                    prediction | ground_truth
                ).sum().item()

                iou = (
                    intersection / union
                    if union
                    else 1.0
                )

                scores.append(iou)

        score = float(np.mean(scores))

        print(
            f"threshold={threshold:.2f} "
            f"IoU={score:.4f}"
        )

        if score > best_iou:
            best_iou = score
            best_threshold = float(threshold)

    save_threshold(best_threshold)

    print(
        f"Best threshold: {best_threshold:.2f}, "
        f"IoU: {best_iou:.4f}"
    )


if __name__ == "__main__":
    main()
