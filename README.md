# GhostNet SSS Detection Prototype

DeepLabV3 + ResNet50 prototype aligned with the GhostNetZero paper.

Dataset: `rehan9599/drishti-sss`.

IMPORTANT: DRISHTI provides YOLO bounding boxes and its `ghost_net` class is synthetic. This repo converts ghost-net boxes into rectangular proxy masks only to make a segmentation prototype runnable. These are NOT real pixel-level ground-truth masks and results must not be presented as real-world ghost-net segmentation accuracy.

## Pipeline

SSS image -> Lee filter + CLAHE -> DeepLabV3-ResNet50 -> pixel probability -> validation-tuned threshold -> binary mask -> masked/overlay image.

## Windows PowerShell

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Download:
```powershell
& ".\.venv\Scripts\python.exe" scripts\download_dataset.py
```

Prepare:
```powershell
& ".\.venv\Scripts\python.exe" scripts\prepare_dataset.py
```

Train:
```powershell
& ".\.venv\Scripts\python.exe" scripts\train.py --epochs 30 --batch-size 4
```

Tune threshold:
```powershell
& ".\.venv\Scripts\python.exe" scripts\tune_threshold.py
```

Evaluate:
```powershell
& ".\.venv\Scripts\python.exe" scripts\evaluate.py
```

Single image:
```powershell
& ".\.venv\Scripts\python.exe" scripts\predict.py --image "path\to\image.jpg"
```

API:
```powershell
& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --reload
```
Then open http://127.0.0.1:8000/docs.

Model checkpoint: `models/deeplabv3_resnet50_best.pt`
Threshold: `models/threshold.json`
Outputs: `outputs/`

For the final SIH system, replace proxy masks with real annotated ghost-net SSS masks. Keep geolocation separate: detection + sonar/GPS metadata + geometry -> lat/lon/depth.
