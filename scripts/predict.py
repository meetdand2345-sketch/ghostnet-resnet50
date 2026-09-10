import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model import build_model
from src.preprocessing import preprocess


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--weights",
        default="models/deeplabv3_resnet50_best.pt"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
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

    model = build_model(True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    size = checkpoint.get("size", 512)

    image = cv2.imread(args.image)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {args.image}"
        )

    original_h, original_w = image.shape[:2]

    # Use the preprocessing function exactly as defined
    # in src/preprocessing.py.
    processed = preprocess(image)

    # Convert to tensor and resize to the model input size.
    if isinstance(processed, np.ndarray):
        tensor = torch.from_numpy(processed)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim == 3 and tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)

        tensor = tensor.float()

        if tensor.max() > 1:
            tensor = tensor / 255.0
    else:
        tensor = processed

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False
    )

    # DeepLabV3 expects 3 channels.
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)

    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)["out"]
        probability = torch.sigmoid(output)[0, 0]

    probability = probability.cpu().numpy()

    mask = (
        probability >= args.threshold
    ).astype(np.uint8) * 255

    mask = cv2.resize(
        mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    probability = cv2.resize(
        probability,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR
    )

    Path("outputs").mkdir(exist_ok=True)

    stem = Path(args.image).stem

    mask_path = f"outputs/mask_{stem}.png"
    masked_path = f"outputs/masked_{stem}.png"
    overlay_path = f"outputs/overlay_{stem}.jpg"

    cv2.imwrite(mask_path, mask)

    masked = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    cv2.imwrite(masked_path, masked)

    overlay = image.copy()

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    cv2.imwrite(overlay_path, overlay)

    max_probability = float(probability.max())
    mask_ratio = float((mask > 0).mean())

    print("Image:", args.image)
    print("Threshold:", args.threshold)
    print("Max probability:", round(max_probability, 4))
    print("Mask area:", round(mask_ratio, 4))
    print("Detected:", mask_ratio > 0)
    print("Mask:", mask_path)
    print("Masked:", masked_path)
    print("Overlay:", overlay_path)


if __name__ == "__main__":
    main()
