import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.dataset import SonarMaskDataset
from src.model import build_model

def dice_loss(logits,target):
    p=torch.sigmoid(logits); inter=(p*target).sum((1,2,3)); den=p.sum((1,2,3))+target.sum((1,2,3))
    return (1-(2*inter+1e-6)/(den+1e-6)).mean()
def val_loss(model,loader,device):
    model.eval(); s=0
    with torch.no_grad():
        for x,y in loader:
            o=model(x.to(device)); loss=.5*F.binary_cross_entropy_with_logits(o['out'],y.to(device))+.5*dice_loss(o['out'],y.to(device)); s+=loss.item()
    return s/max(1,len(loader))
def main():
    a=argparse.ArgumentParser(); a.add_argument('--epochs',type=int,default=30); a.add_argument('--batch-size',type=int,default=4); a.add_argument('--lr',type=float,default=1e-4); a.add_argument('--size',type=int,default=512); z=a.parse_args()
    d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',d)
    tr=SonarMaskDataset('data/ghostnet/train/images','data/ghostnet/train/masks',a.size); va=SonarMaskDataset('data/ghostnet/val/images','data/ghostnet/val/masks',a.size)
    tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=0); vl=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=0)
    model=build_model(True).to(d); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); best=1e9; Path('models').mkdir(exist_ok=True)
    for e in range(1,a.epochs+1):
        model.train(); s=0
        for x,y in tl:
            x,y=x.to(d),y.to(d); opt.zero_grad(); o=model(x); loss=.5*F.binary_cross_entropy_with_logits(o['out'],y)+.5*dice_loss(o['out'],y)
            if o.get('aux') is not None: loss=loss+.4*F.binary_cross_entropy_with_logits(o['aux'],y)
            loss.backward(); opt.step(); s+=loss.item()
        v=val_loss(model,vl,d); print(f'Epoch {e}/{a.epochs} train={s/max(1,len(tl)):.4f} val={v:.4f}')
        if v<best:
            best=v; torch.save({'model_state_dict':model.state_dict(),'size':a.size},'models/deeplabv3_resnet50_best.pt'); print('saved best')
if __name__=='__main__': main()
