from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset

class SonarMaskDataset(Dataset):
    def __init__(self,image_dir,mask_dir,size=512):
        self.image_dir=Path(image_dir); self.mask_dir=Path(mask_dir); self.size=size
        self.images=sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    def __len__(self): return len(self.images)
    def __getitem__(self,idx):
        p=self.images[idx]; m=self.mask_dir/f'{p.stem}.png'
        image=cv2.imread(str(p)); mask=cv2.imread(str(m),cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None: raise RuntimeError(f'Cannot read {p} or {m}')
        image=cv2.resize(image,(self.size,self.size),interpolation=cv2.INTER_AREA)
        mask=cv2.resize(mask,(self.size,self.size),interpolation=cv2.INTER_NEAREST)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        mean=np.array([.485,.456,.406],np.float32); std=np.array([.229,.224,.225],np.float32)
        image=(image-mean)/std
        image=torch.from_numpy(image.transpose(2,0,1)).float()
        mask=torch.from_numpy((mask>127).astype(np.float32)).unsqueeze(0)
        return image,mask
