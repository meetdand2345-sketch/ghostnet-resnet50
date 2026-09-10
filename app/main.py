import base64,io
from pathlib import Path
import cv2,numpy as np,torch
from fastapi import FastAPI,File,HTTPException,UploadFile
from PIL import Image
from src.model import build_model
from src.preprocessing import preprocess
from src.utils import load_threshold
app=FastAPI(title='GhostNet SSS Detector',version='1.0.0')
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); WEIGHTS=Path('models/deeplabv3_resnet50_best.pt'); MODEL=None
if WEIGHTS.exists():
    MODEL=build_model(False); MODEL.load_state_dict(torch.load(WEIGHTS,map_location=DEVICE)['model_state_dict']); MODEL.to(DEVICE).eval()
THRESHOLD=load_threshold()
def enc(im):
    ok,b=cv2.imencode('.png',im)
    if not ok: raise RuntimeError('PNG encoding failed')
    return base64.b64encode(b.tobytes()).decode()
@app.get('/')
def root(): return {'service':'GhostNet SSS Detector','model':'DeepLabV3-ResNet50','model_loaded':MODEL is not None,'threshold':THRESHOLD}
@app.post('/predict')
async def predict(file:UploadFile=File(...)):
    if MODEL is None: raise HTTPException(503,'Model checkpoint not found. Train first.')
    try: pil=Image.open(io.BytesIO(await file.read())).convert('RGB')
    except Exception as e: raise HTTPException(400,'Invalid image') from e
    orig=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR); proc=cv2.resize(preprocess(orig),(512,512))
    rgb=cv2.cvtColor(proc,cv2.COLOR_BGR2RGB).astype(np.float32)/255; rgb=(rgb-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
    x=torch.from_numpy(rgb.transpose(2,0,1)).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad(): prob=torch.sigmoid(MODEL(x)['out'])[0,0].cpu().numpy()
    mask=cv2.resize((prob>=THRESHOLD).astype(np.uint8)*255,(orig.shape[1],orig.shape[0]),interpolation=cv2.INTER_NEAREST)
    masked=cv2.bitwise_and(orig,orig,mask=mask); overlay=orig.copy(); fg=mask>0; overlay[fg]=(0.55*overlay[fg]+0.45*np.array([0,0,255])).astype(np.uint8)
    return {'ghost_net_detected':bool(mask.any()),'threshold':THRESHOLD,'max_pixel_probability':round(float(prob.max()),4),'top_1pct_probability':round(float(np.percentile(prob,99)),4),'mask_area_ratio':round(float((mask>0).mean()),4),'width':orig.shape[1],'height':orig.shape[0],'mask_png_base64':enc(mask),'masked_image_png_base64':enc(masked),'overlay_png_base64':enc(overlay)}
