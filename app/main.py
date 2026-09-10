from pathlib import Path
import base64

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.model import build_model
from src.preprocessing import preprocess


app = FastAPI(title="GhostNet ResNet50 API")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = Path("models/deeplabv3_resnet50_best.pt")

# IMPORTANT:
# The checkpoint was trained with the auxiliary classifier enabled.
MODEL = build_model(True)

checkpoint = torch.load(WEIGHTS, map_location=DEVICE)
MODEL.load_state_dict(checkpoint["model_state_dict"])
MODEL.to(DEVICE)
MODEL.eval()

THRESHOLD = 0.5


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "DeepLabV3-ResNet50",
        "device": str(DEVICE),
        "threshold": THRESHOLD,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": True,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()

    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not decode image"},
        )

    original_h, original_w = image.shape[:2]

    # Preprocess
    x = preprocess(image)

    # Make sure tensor is CHW
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if x.ndim == 3 and x.shape[0] not in (1, 3):
        x = x.permute(2, 0, 1)

    # Ensure 3 channels
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)

    x = x.float().unsqueeze(0)

    # Model expects 512x512
    x = torch.nn.functional.interpolate(
        x,
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )

    x = x.to(DEVICE)

    # Inference
    with torch.no_grad():
        output = MODEL(x)["out"]
        probability = torch.sigmoid(output)[0, 0]

    # Restore original image dimensions
    probability = torch.nn.functional.interpolate(
        probability.unsqueeze(0).unsqueeze(0),
        size=(original_h, original_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()

    mask = (probability >= THRESHOLD).astype(np.uint8) * 255

    detected = bool(mask.max() > 0)

    # Encode mask
    _, buffer = cv2.imencode(".png", mask)
    mask_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(
        {
            "detected": detected,
            "confidence": float(probability.max()),
            "mask_area_pixels": int(np.count_nonzero(mask)),
            "image_width": original_w,
            "image_height": original_h,
            "threshold": THRESHOLD,
            "mask_base64": mask_base64,
        }
    )