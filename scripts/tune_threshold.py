import argparse, numpy as np, torch
from torch.utils.data import DataLoader
from src.dataset import SonarMaskDataset
from src.model import build_model
from src.utils import save_threshold
def main():
    a=argparse.ArgumentParser(); a.add_argument('--weights',default='models/deeplabv3_resnet50_best.pt'); a.add_argument('--size',type=int,default=512); z=a.parse_args()
    d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=build_model(False); m.load_state_dict(torch.load(z.weights,map_location=d)['model_state_dict']); m.to(d).eval()
    ds=SonarMaskDataset('data/ghostnet/val/images','data/ghostnet/val/masks',z.size); dl=DataLoader(ds,batch_size=4,num_workers=0)
    cache=[]
    with torch.no_grad():
        for x,y in dl: cache.append((torch.sigmoid(m(x.to(d))['out']).cpu(),y))
    best=(-1,.5)
    for t in np.arange(.1,.91,.05):
        scores=[]
        for p,y in cache:
            for pp,yy in zip(p,y):
                pr=pp>=t; gt=yy>=.5; inter=(pr&gt).sum().item(); union=(pr|gt).sum().item(); scores.append(inter/union if union else 1)
        score=float(np.mean(scores)); print(f'threshold={t:.2f} IoU={score:.4f}')
        if score>best[0]: best=(score,float(t))
    save_threshold(best[1]); print(f'Best threshold: {best[1]:.2f}, IoU: {best[0]:.4f}')
if __name__=='__main__': main()
