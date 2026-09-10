import argparse,cv2,numpy as np,torch
from pathlib import Path
from src.model import build_model
from src.preprocessing import preprocess
from src.utils import load_threshold
def main():
    a=argparse.ArgumentParser(); a.add_argument('--image',required=True); a.add_argument('--weights',default='models/deeplabv3_resnet50_best.pt'); a.add_argument('--size',type=int,default=512); z=a.parse_args()
    d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=build_model(False); m.load_state_dict(torch.load(z.weights,map_location=d)['model_state_dict']); m.to(d).eval()
    orig=cv2.imread(z.image)
    if orig is None: raise FileNotFoundError(z.image)
    im=cv2.resize(preprocess(orig),(z.size,z.size)); rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    rgb=(rgb-np.array([.485,.456,.406]))/np.array([.229,.224,.225]); x=torch.from_numpy(rgb.transpose(2,0,1)).float().unsqueeze(0).to(d)
    with torch.no_grad(): prob=torch.sigmoid(m(x)['out'])[0,0].cpu().numpy()
    t=load_threshold(); small=(prob>=t).astype(np.uint8)*255; mask=cv2.resize(small,(orig.shape[1],orig.shape[0]),interpolation=cv2.INTER_NEAREST)
    masked=cv2.bitwise_and(orig,orig,mask=mask); overlay=orig.copy(); fg=mask>0; overlay[fg]=(0.55*overlay[fg]+0.45*np.array([0,0,255])).astype(np.uint8)
    out=Path('outputs'); out.mkdir(exist_ok=True); stem=Path(z.image).stem
    cv2.imwrite(str(out/f'mask_{stem}.png'),mask); cv2.imwrite(str(out/f'masked_{stem}.png'),masked); cv2.imwrite(str(out/f'overlay_{stem}.jpg'),overlay)
    print('Net detected:',bool(mask.any())); print(f'Threshold: {t:.4f}'); print(f'Max probability: {prob.max():.4f}'); print(f'Top 1% probability: {np.percentile(prob,99):.4f}'); print(f'Mask area ratio: {(mask>0).mean():.4f}'); print('Saved:',out.resolve())
if __name__=='__main__': main()
