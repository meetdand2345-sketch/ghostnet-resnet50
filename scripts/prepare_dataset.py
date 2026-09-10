from pathlib import Path
import shutil, cv2, numpy as np
RAW=Path('data/raw'); OUT=Path('data/ghostnet'); GHOST_CLASS_ID=3
def prepare(split):
    imgs=RAW/split/'images'; labels=RAW/split/'labels'
    if not imgs.exists(): raise FileNotFoundError(f'Missing {imgs}. Finish download first.')
    oi=OUT/split/'images'; om=OUT/split/'masks'; oi.mkdir(parents=True,exist_ok=True); om.mkdir(parents=True,exist_ok=True)
    total=positive=boxes=0
    for p in sorted(imgs.iterdir()):
        if p.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
        im=cv2.imread(str(p)); h,w=im.shape[:2]; mask=np.zeros((h,w),np.uint8)
        lp=labels/f'{p.stem}.txt'
        if lp.exists():
            for line in lp.read_text().splitlines():
                q=line.split()
                if len(q)!=5: continue
                cls,xc,yc,bw,bh=map(float,q)
                if int(cls)!=GHOST_CLASS_ID: continue
                x1=max(0,int((xc-bw/2)*w)); y1=max(0,int((yc-bh/2)*h))
                x2=min(w-1,int((xc+bw/2)*w)); y2=min(h-1,int((yc+bh/2)*h))
                if x2>x1 and y2>y1: mask[y1:y2+1,x1:x2+1]=255; boxes+=1
        shutil.copy2(p,oi/p.name); cv2.imwrite(str(om/f'{p.stem}.png'),mask)
        total+=1; positive+=int(mask.any())
    print(f'{split}: {total} images | {positive} positive | {boxes} proxy ghost-net boxes')
def main():
    for s in ('train','val','test'): prepare(s)
    print('WARNING: proxy rectangular masks, not true segmentation labels.')
if __name__=='__main__': main()
