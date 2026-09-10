import argparse,torch
from torch.utils.data import DataLoader
from src.dataset import SonarMaskDataset
from src.model import build_model
from src.utils import load_threshold
def main():
    a=argparse.ArgumentParser(); a.add_argument('--weights',default='models/deeplabv3_resnet50_best.pt'); a.add_argument('--size',type=int,default=512); z=a.parse_args()
    d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=build_model(False); m.load_state_dict(torch.load(z.weights,map_location=d)['model_state_dict']); m.to(d).eval()
    ds=SonarMaskDataset('data/ghostnet/test/images','data/ghostnet/test/masks',z.size); dl=DataLoader(ds,batch_size=4,num_workers=0); t=load_threshold()
    tp=fp=fn=0
    with torch.no_grad():
        for x,y in dl:
            p=torch.sigmoid(m(x.to(d))['out']).cpu()>=t; g=y>=.5
            tp+=(p&g).sum().item(); fp+=(p&~g).sum().item(); fn+=(~p&g).sum().item()
    print(f'Threshold: {t:.4f}'); print(f'IoU: {tp/max(tp+fp+fn,1):.4f}'); print(f'Dice: {2*tp/max(2*tp+fp+fn,1):.4f}'); print(f'Precision: {tp/max(tp+fp,1):.4f}'); print(f'Recall: {tp/max(tp+fn,1):.4f}')
if __name__=='__main__': main()
