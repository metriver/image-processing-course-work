"""
运行实验三：U-Net + Attention U-Net 训练+测试+评估
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np, random
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.datasets import check_drive, load_drive_data
from utils.metrics import compute_binary_metrics, print_metrics_table, plot_roc_curve, compute_roc_metrics, plot_loss_curve
from exp3_deep_learning_vessel import UNet, AttentionUNet, BCEDiceLoss

OUT = "outputs/exp3"; MDL = "models"
os.makedirs(OUT, exist_ok=True); os.makedirs(MDL, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PS, BS, EPOCHS, LR = 48, 64, 50, 0.001

class PatchDataset(Dataset):
    def __init__(self, imgs, gts, mks, ps=48, stride=10, aug=True):
        self.ps = ps; self.aug = aug; self.patches = []
        half = ps//2
        for idx in range(len(imgs)):
            _, img = imgs[idx]; gt = gts[idx]; mk = mks[idx]
            h, w = img.shape[:2]
            for y in range(half, h-half, stride):
                for x in range(half, w-half, stride):
                    if mk[y, x] == 0: continue
                    y0,y1 = y-half,y+half; x0,x1 = x-half,x+half
                    self.patches.append((img[y0:y1,x0:x1].copy(), gt[y0:y1,x0:x1].copy(), mk[y0:y1,x0:x1].copy()))
        print(f"  {len(self.patches)} patches (stride={stride})")

    def __len__(self): return len(self.patches)
    def __getitem__(self, idx):
        ip, gp, mp = self.patches[idx]
        if self.aug and random.random() > 0.5:
            t = random.randint(0,3)
            if t==0: ip,gp = np.fliplr(ip).copy(), np.fliplr(gp).copy()
            elif t==1: ip,gp = np.flipud(ip).copy(), np.flipud(gp).copy()
            elif t==2: k=random.randint(1,3); ip=np.rot90(ip,k).copy(); gp=np.rot90(gp,k).copy()
            else: ip=np.clip(random.uniform(0.8,1.2)*ip+random.randint(-10,10),0,255).astype(np.uint8)
        it = torch.from_numpy(ip.astype(np.float32)).permute(2,0,1)/255.0
        return it, torch.from_numpy((gp>127).astype(np.float32)).unsqueeze(0), torch.from_numpy((mp>0).astype(np.float32)).unsqueeze(0)

def tr_epoch(m, dl, crit, opt):
    m.train(); s=0
    for im,gt,mk in dl:
        im,gt,mk=im.to(DEV),gt.to(DEV),mk.to(DEV); opt.zero_grad()
        l=crit(m(im)*mk, gt*mk); l.backward(); opt.step(); s+=l.item()*im.size(0)
    return s/len(dl.dataset)

def vl_epoch(m, dl, crit):
    m.eval(); s=0
    with torch.no_grad():
        for im,gt,mk in dl:
            im,gt,mk=im.to(DEV),gt.to(DEV),mk.to(DEV)
            l=crit(m(im)*mk, gt*mk); s+=l.item()*im.size(0)
    return s/len(dl.dataset)

def train(m, tdl, vdl, nm):
    m=m.to(DEV); crit=BCEDiceLoss(0.5); opt=optim.Adam(m.parameters(), lr=LR, weight_decay=1e-5)
    sched=ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    hist={'train':[],'val':[]}; bv=float('inf')
    print(f"训练 [{nm}] {EPOCHS} epochs")
    for ep in range(1, EPOCHS+1):
        tl=tr_epoch(m, tdl, crit, opt); vl=vl_epoch(m, vdl, crit)
        hist['train'].append(tl); hist['val'].append(vl); sched.step(vl)
        if vl<bv: bv=vl; torch.save(m.state_dict(), f"{MDL}/{nm}_best.pth")
        if ep%10==0: print(f"  Epoch {ep:3d}: TL={tl:.4f} VL={vl:.4f}")
    m.load_state_dict(torch.load(f"{MDL}/{nm}_best.pth", map_location=DEV, weights_only=True))
    return m, hist

@torch.no_grad()
def predict(m, img):
    m.eval(); h,w=img.shape[:2]; half=PS//2
    it=torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(DEV)/255.0
    prob=np.zeros((h,w),np.float64); cnt=np.zeros((h,w),np.float64)
    for y in range(half, h-half, PS):
        for x in range(half, w-half, PS):
            y0,y1=y-half,y+half; x0,x1=x-half,x+half
            p=m(it[:,:,y0:y1,x0:x1]).cpu().numpy()[0,0]
            prob[y0:y1,x0:x1]+=p; cnt[y0:y1,x0:x1]+=1
    prob[cnt>0]/=cnt[cnt>0]; return prob

def evaluate(m, t_imgs, t_gts, t_mks, nm):
    m.eval(); ml=[]; ps=[]
    for (nn,img),gt,mk in tqdm(zip(t_imgs,t_gts,t_mks), total=len(t_imgs), desc=f"Test {nm}"):
        pm=predict(m,img); pb=(pm>0.5).astype(np.uint8)*255
        ps.append((nn,pm)); ml.append(compute_binary_metrics(pb,gt,mk))
    avg={}
    for k in ['accuracy','sensitivity','specificity','precision','f1','dice']:
        avg[k]=np.mean([x[k] for x in ml])
    print_metrics_table(avg, f"{nm} 平均指标"); return ml, avg, ps

def vis(t_imgs, t_gts, ps, nm, num=5):
    for idx in range(min(num, len(t_imgs))):
        n,img=t_imgs[idx]; _,pm=ps[idx]; pb=(pm>0.5).astype(np.uint8)*255; gt_b=(t_gts[idx]>127).astype(np.uint8)
        fig,axes=plt.subplots(2,3,figsize=(14,9))
        axes[0,0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); axes[0,0].set_title("Original"); axes[0,0].axis('off')
        im=axes[0,1].imshow(pm, cmap='hot', vmin=0, vmax=1); axes[0,1].set_title("Probability"); axes[0,1].axis('off')
        plt.colorbar(im, ax=axes[0,1], fraction=0.046)
        axes[0,2].imshow(t_gts[idx], cmap='gray'); axes[0,2].set_title("GT"); axes[0,2].axis('off')
        axes[1,0].imshow(pb, cmap='gray'); axes[1,0].set_title("Pred"); axes[1,0].axis('off')
        d=np.zeros((*pb.shape,3),np.uint8)
        d[:,:,1]=(pb>127)&gt_b*255; d[:,:,2]=(pb>127)&(1-gt_b)*255; d[:,:,0]=((pb<=127)&gt_b)*255
        axes[1,1].imshow(d); axes[1,1].set_title("Diff(G=TP,R=FP,B=FN)"); axes[1,1].axis('off')
        ov=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).copy(); ov[pb>0]=[0,255,0]
        axes[1,2].imshow(ov); axes[1,2].set_title("Overlay"); axes[1,2].axis('off')
        plt.suptitle(f"{nm} - {n}", fontsize=12); plt.tight_layout()
        plt.savefig(f"{OUT}/{nm}_{os.path.splitext(n)[0]}.png", dpi=150, bbox_inches='tight'); plt.close()

def main():
    if not check_drive("datasets/DRIVE"): print("DRIVE not found!"); return
    tr_imgs,tr_gts,tr_mks=load_drive_data("datasets/DRIVE",True)
    te_imgs,te_gts,te_mks=load_drive_data("datasets/DRIVE",False)
    a,b,c=tr_imgs[:16],tr_gts[:16],tr_mks[:16]
    d,e,f=tr_imgs[16:],tr_gts[16:],tr_mks[16:]
    print(f"Train:16 Val:4 Test:{len(te_imgs)}")

    tr_ds=PatchDataset(a,b,c,PS,10,True); vl_ds=PatchDataset(d,e,f,PS,30,False)
    tr_dl=DataLoader(tr_ds,BS,shuffle=True,pin_memory=True); vl_dl=DataLoader(vl_ds,BS,shuffle=False,pin_memory=True)

    # U-Net
    print("\n[U-Net]")
    u=UNet(3,1,64); print(f"Params: {sum(p.numel() for p in u.parameters()):,}")
    u,hu=train(u,tr_dl,vl_dl,"unet")
    plot_loss_curve(hu['train'],hu['val'],"U-Net Loss",f"{OUT}/unet_loss.png")
    _,au,pu=evaluate(u,te_imgs,te_gts,te_mks,"UNet"); vis(te_imgs,te_gts,pu,"UNet",5)
    ag=np.concatenate([g.flatten() for g in te_gts]); amk=np.concatenate([m.flatten() for m in te_mks])
    ru=compute_roc_metrics(np.concatenate([p[1].flatten() for p in pu]),ag,amk); au['auc']=ru['auc']
    plot_roc_curve(ru,"U-Net ROC",f"{OUT}/unet_roc.png")

    # Attention U-Net
    print("\n[Attention U-Net]")
    au_net=AttentionUNet(3,1,64); print(f"Params: {sum(p.numel() for p in au_net.parameters()):,}")
    au_net,ha=train(au_net,tr_dl,vl_dl,"attention_unet")
    plot_loss_curve(ha['train'],ha['val'],"Attention U-Net Loss",f"{OUT}/attention_unet_loss.png")
    _,aa,pa=evaluate(au_net,te_imgs,te_gts,te_mks,"Attention UNet"); vis(te_imgs,te_gts,pa,"Attention_UNet",5)
    ra=compute_roc_metrics(np.concatenate([p[1].flatten() for p in pa]),ag,amk); aa['auc']=ra['auc']
    plot_roc_curve(ra,"Attention U-Net ROC",f"{OUT}/attention_unet_roc.png")

    # 对比
    print(f"\n{'Method':<25s} {'Acc':>8s} {'Se':>8s} {'Sp':>8s} {'F1':>8s} {'AUC':>8s}")
    print("-"*70)
    for mn,mv in [("Unsupervised",{'accuracy':0.8745,'sensitivity':0.1707,'specificity':0.9772,'f1':0.2501,'auc':0.5740}),
                  ("U-Net",au), ("Attn U-Net",aa)]:
        print(f"{mn:<25s} {mv['accuracy']:8.4f} {mv['sensitivity']:8.4f} {mv['specificity']:8.4f} {mv['f1']:8.4f} {mv['auc']:8.4f}")

    fig,ax=plt.subplots(figsize=(8,6))
    ax.plot(ru['fpr'],ru['tpr'],label=f"U-Net AUC={ru['auc']:.4f}")
    ax.plot(ra['fpr'],ra['tpr'],label=f"Attn U-Net AUC={ra['auc']:.4f}")
    ax.plot([0,1],[0,1],'k--',label="Random"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Comparison"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/roc_comparison.png",dpi=150,bbox_inches='tight'); plt.close()

    fig,ax=plt.subplots(figsize=(10,6))
    mn_n=['Acc','Se','Sp','F1','AUC']; x=np.arange(5); w=0.25
    vu=[au.get(k,0) for k in ['accuracy','sensitivity','specificity','f1','auc']]
    va=[aa.get(k,0) for k in ['accuracy','sensitivity','specificity','f1','auc']]
    ax.bar(x-w/2,vu,w,label='U-Net',color='steelblue'); ax.bar(x+w/2,va,w,label='Attn U-Net',color='coral')
    ax.set_xticks(x); ax.set_xticklabels(mn_n); ax.set_ylim(0,1.05); ax.set_title('DL Segmentation')
    ax.legend(); ax.grid(axis='y',alpha=0.3)
    for i,(u,a) in enumerate(zip(vu,va)):
        ax.text(i-w/2,u+0.01,f'{u:.3f}',ha='center',va='bottom',fontsize=8)
        ax.text(i+w/2,a+0.01,f'{a:.3f}',ha='center',va='bottom',fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/metrics_comparison.png",dpi=150,bbox_inches='tight'); plt.close()

    # 分割对比图
    for idx in range(min(3,len(te_imgs))):
        nm,img=te_imgs[idx]; gt_b=(te_gts[idx]>127).astype(np.uint8)
        _,ppu=pu[idx]; pb_u=(ppu>0.5).astype(np.uint8)*255
        _,ppa=pa[idx]; pb_a=(ppa>0.5).astype(np.uint8)*255
        fig,axes=plt.subplots(2,4,figsize=(16,8))
        axes[0,0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); axes[0,0].set_title("Original"); axes[0,0].axis('off')
        axes[0,1].imshow(te_gts[idx],cmap='gray'); axes[0,1].set_title("GT"); axes[0,1].axis('off')
        axes[0,2].imshow(pb_u,cmap='gray'); axes[0,2].set_title("U-Net"); axes[0,2].axis('off')
        axes[0,3].imshow(pb_a,cmap='gray'); axes[0,3].set_title("Attn U-Net"); axes[0,3].axis('off')
        def df(pb,gt):
            d=np.zeros((*pb.shape,3),np.uint8)
            d[:,:,1]=(pb>127)&gt*255; d[:,:,2]=(pb>127)&(1-gt)*255; d[:,:,0]=((pb<=127)&gt)*255
            return d
        axes[1,0].imshow(df(pb_u,gt_b)); axes[1,0].set_title("U-Net Diff"); axes[1,0].axis('off')
        axes[1,1].imshow(df(pb_a,gt_b)); axes[1,1].set_title("Attn Diff"); axes[1,1].axis('off')
        ovu=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).copy(); ovu[pb_u>0]=[0,255,0]
        axes[1,2].imshow(ovu); axes[1,2].set_title("U-Net Overlay"); axes[1,2].axis('off')
        ova=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).copy(); ova[pb_a>0]=[0,255,0]
        axes[1,3].imshow(ova); axes[1,3].set_title("Attn Overlay"); axes[1,3].axis('off')
        plt.suptitle(f"Comparison - {nm}",fontsize=12); plt.tight_layout()
        plt.savefig(f"{OUT}/compare_{os.path.splitext(nm)[0]}.png",dpi=150,bbox_inches='tight'); plt.close()

    with open(f"{OUT}/metrics.txt",'w') as f:
        f.write("Method,Accuracy,Sensitivity,Specificity,F1,AUC\n")
        for mn,mv in [("Unsupervised",{'accuracy':0.8745,'sensitivity':0.1707,'specificity':0.9772,'f1':0.2501,'auc':0.5740}),
                      ("U-Net",au), ("Attention U-Net",aa)]:
            f.write(f"{mn},{mv['accuracy']:.4f},{mv['sensitivity']:.4f},{mv['specificity']:.4f},{mv['f1']:.4f},{mv['auc']:.4f}\n")
    print(f"\nDone -> {OUT}/")

if __name__=="__main__": main()
