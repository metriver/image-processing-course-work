"""
实验三：深度学习眼底血管分割
U-Net + Attention U-Net（扩展），BCE+Dice Loss
"""
import os, sys, cv2, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.datasets import check_drive, load_drive_data, auto_download_drive, DRIVE_INFO
from utils.metrics import compute_binary_metrics, print_metrics_table, plot_roc_curve, compute_roc_metrics, plot_loss_curve

CONFIG = {
    'patch_size': 48, 'stride': 10, 'test_stride': 48,
    'batch_size': 64, 'epochs': 50, 'lr': 0.001, 'weight_decay': 1e-5,
    'base_channels': 64, 'num_workers': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'outputs/exp3', 'model_dir': 'models',
}
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['model_dir'], exist_ok=True)


class DoubleConv(nn.Module):
    """Conv3x3->BN->ReLU->Conv3x3->BN->ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x): return self.conv(x)


class Down(nn.Module):
    """MaxPool2x2 + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x): return self.down(x)


class Up(nn.Module):
    """转置卷积上采样 + skip cat + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """标准 U-Net：4层编码-解码 + 跳跃连接"""
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = Down(c, c*2); self.enc3 = Down(c*2, c*4); self.enc4 = Down(c*4, c*8)
        self.bottleneck = DoubleConv(c*8, c*8)
        self.dec4 = Up(c*8, c*4); self.dec3 = Up(c*4, c*2); self.dec2 = Up(c*2, c)
        self.out = nn.Conv2d(c, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.dec4(b, e3); d3 = self.dec3(d4, e2); d2 = self.dec2(d3, e1)
        return torch.sigmoid(self.out(d2))


class AttentionGate(nn.Module):
    """注意力门控：g（解码信号）+ x（跳跃特征）→ 空间注意力系数"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        return x * self.psi(F.relu(self.W_g(g) + self.W_x(x), inplace=True))


class AttentionUp(nn.Module):
    """上采样 + AttentionGate + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.attn = AttentionGate(out_ch, out_ch, out_ch // 2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([self.attn(g=x, x=skip), x], dim=1))


class AttentionUNet(nn.Module):
    """Attention U-Net：跳跃连接中加入注意力门控"""
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = Down(c, c*2); self.enc3 = Down(c*2, c*4); self.enc4 = Down(c*4, c*8)
        self.bottleneck = DoubleConv(c*8, c*8)
        self.dec4 = AttentionUp(c*8, c*4); self.dec3 = AttentionUp(c*4, c*2); self.dec2 = AttentionUp(c*2, c)
        self.out = nn.Conv2d(c, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.dec4(b, e3); d3 = self.dec3(d4, e2); d2 = self.dec2(d3, e1)
        return torch.sigmoid(self.out(d2))


class BCEDiceLoss(nn.Module):
    """0.5 * BCE + 0.5 * Dice"""
    def __init__(self, alpha=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha; self.smooth = smooth; self.bce = nn.BCELoss()

    def forward(self, p, t):
        bce = self.bce(p, t)
        pf, tf = p.view(-1), t.view(-1)
        inter = (pf * tf).sum()
        dice = (2*inter + self.smooth) / (pf.sum() + tf.sum() + self.smooth)
        return self.alpha * bce + (1 - self.alpha) * (1 - dice)


class PatchDataset(Dataset):
    """从眼底图像提取固定大小 patch，FOV 内有效才保留"""
    def __init__(self, images, gts, masks, patch_size=48, stride=10, augment=True):
        self.ps = patch_size; self.aug = augment
        self.patches = []
        half = patch_size // 2
        for idx in range(len(images)):
            _, img = images[idx]; gt = gts[idx]; mask = masks[idx]
            h, w = img.shape[:2]
            for y in range(half, h - half, stride):
                for x in range(half, w - half, stride):
                    if mask[y, x] == 0: continue
                    y0, y1 = y-half, y+half; x0, x1 = x-half, x+half
                    self.patches.append((img[y0:y1, x0:x1].copy(), gt[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()))
        print(f"提取 {len(self.patches)} 个 patch (stride={stride})")

    def __len__(self): return len(self.patches)

    def __getitem__(self, idx):
        ip, gp, mp = self.patches[idx]
        if self.aug and random.random() > 0.5:
            t = random.randint(0, 3)
            if t == 0: ip, gp = np.fliplr(ip).copy(), np.fliplr(gp).copy()
            elif t == 1: ip, gp = np.flipud(ip).copy(), np.flipud(gp).copy()
            elif t == 2: k = random.randint(1, 3); ip = np.rot90(ip, k).copy(); gp = np.rot90(gp, k).copy()
            else: ip = np.clip(random.uniform(0.8,1.2)*ip + random.randint(-10,10), 0, 255).astype(np.uint8)
        it = torch.from_numpy(ip.astype(np.float32)).permute(2, 0, 1) / 255.0
        gt_t = torch.from_numpy((gp > 127).astype(np.float32)).unsqueeze(0)
        mt = torch.from_numpy((mp > 0).astype(np.float32)).unsqueeze(0)
        return it, gt_t, mt


def train_epoch(model, dl, criterion, opt, device):
    model.train(); s = 0
    for im, gt, mk in dl:
        im, gt, mk = im.to(device), gt.to(device), mk.to(device)
        opt.zero_grad()
        l = criterion(model(im) * mk, gt * mk)
        l.backward(); opt.step(); s += l.item() * im.size(0)
    return s / len(dl.dataset)


def val_epoch(model, dl, criterion, device):
    model.eval(); s = 0
    with torch.no_grad():
        for im, gt, mk in dl:
            im, gt, mk = im.to(device), gt.to(device), mk.to(device)
            l = criterion(model(im) * mk, gt * mk); s += l.item() * im.size(0)
    return s / len(dl.dataset)


def train_model(model, train_dl, val_dl, name, config):
    device = config['device']; model = model.to(device)
    criterion = BCEDiceLoss(0.5)
    opt = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    hist = {'train': [], 'val': []}; best_vl = float('inf')
    print(f"训练 [{name}] {config['epochs']} epochs")
    for ep in range(1, config['epochs'] + 1):
        tl = train_epoch(model, train_dl, criterion, opt, device)
        vl = val_epoch(model, val_dl, criterion, device)
        hist['train'].append(tl); hist['val'].append(vl)
        sched.step(vl)
        if vl < best_vl:
            best_vl = vl; torch.save(model.state_dict(), f"{config['model_dir']}/{name}_best.pth")
        if ep % 10 == 0: print(f"  Epoch {ep:3d}: TrainLoss={tl:.4f} ValLoss={vl:.4f}")
    model.load_state_dict(torch.load(f"{config['model_dir']}/{name}_best.pth", map_location=device, weights_only=True))
    return model, hist


@torch.no_grad()
def predict_full(model, image, device, patch_size=48, stride=48):
    """滑动窗口推理，取所有覆盖 patch 的均值"""
    model.eval()
    h, w = image.shape[:2]; half = patch_size // 2
    it = torch.from_numpy(image.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device) / 255.0
    prob = np.zeros((h, w), dtype=np.float64); cnt = np.zeros((h, w), dtype=np.float64)
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            y0, y1 = y-half, y+half; x0, x1 = x-half, x+half
            p = model(it[:,:,y0:y1,x0:x1]).cpu().numpy()[0,0]
            prob[y0:y1, x0:x1] += p; cnt[y0:y1, x0:x1] += 1
    prob[cnt > 0] /= cnt[cnt > 0]
    return prob


def evaluate_model(model, test_images, test_gts, test_masks, config, name):
    device = config['device']; model.eval()
    mlist, preds = [], []
    for (nm, img), gt, mask in tqdm(zip(test_images, test_gts, test_masks), total=len(test_images), desc=f"Test {name}"):
        pm = predict_full(model, img, device, config['patch_size'], config['test_stride'])
        pb = (pm > 0.5).astype(np.uint8) * 255
        preds.append((nm, pm))
        mlist.append(compute_binary_metrics(pb, gt, mask))
    avg = {}
    for k in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'dice']:
        avg[k] = np.mean([m[k] for m in mlist])
    print_metrics_table(avg, f"{name} 平均指标")
    return mlist, avg, preds


def visualize_dl_result(test_images, test_gts, preds, name, num=5, out_dir="outputs/exp3"):
    for idx in range(min(num, len(test_images))):
        n, img = test_images[idx]; _, pm = preds[idx]
        pb = (pm > 0.5).astype(np.uint8)*255; gt_b = (test_gts[idx] > 127).astype(np.uint8)
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0,0].set_title("Original"); axes[0,0].axis('off')
        im = axes[0,1].imshow(pm, cmap='hot', vmin=0, vmax=1); axes[0,1].set_title("Probability"); axes[0,1].axis('off')
        plt.colorbar(im, ax=axes[0,1], fraction=0.046)
        axes[0,2].imshow(test_gts[idx], cmap='gray'); axes[0,2].set_title("GT"); axes[0,2].axis('off')
        axes[1,0].imshow(pb, cmap='gray'); axes[1,0].set_title("Prediction"); axes[1,0].axis('off')
        diff = np.zeros((*pb.shape,3), dtype=np.uint8)
        diff[:,:,1]=(pb>127)&gt_b*255; diff[:,:,2]=(pb>127)&(1-gt_b)*255; diff[:,:,0]=((pb<=127)&gt_b)*255
        axes[1,1].imshow(diff); axes[1,1].set_title("Diff(G=TP,R=FP,B=FN)"); axes[1,1].axis('off')
        ov = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy(); ov[pb>0]=[0,255,0]
        axes[1,2].imshow(ov); axes[1,2].set_title("Overlay"); axes[1,2].axis('off')
        plt.suptitle(f"{name} - {n}", fontsize=12); plt.tight_layout()
        plt.savefig(f"{out_dir}/{name}_{os.path.splitext(n)[0]}.png", dpi=150, bbox_inches='tight'); plt.close()


def main():
    print("实验三：深度学习眼底血管分割（含 Attention U-Net 扩展）")
    config = CONFIG
    if not check_drive("datasets/DRIVE"):
        print("DRIVE 数据集未找到！"); auto_download_drive("datasets/DRIVE"); return

    train_imgs, train_gts, train_masks = load_drive_data("datasets/DRIVE", train=True)
    test_imgs, test_gts, test_masks = load_drive_data("datasets/DRIVE", train=False)
    tr_imgs = train_imgs[:16]; tr_gts = train_gts[:16]; tr_masks = train_masks[:16]
    vl_imgs = train_imgs[16:]; vl_gts = train_gts[16:]; vl_masks = train_masks[16:]
    print(f"Train:16, Val:4, Test:{len(test_imgs)}")

    tr_ds = PatchDataset(tr_imgs, tr_gts, tr_masks, config['patch_size'], stride=10, augment=True)
    vl_ds = PatchDataset(vl_imgs, vl_gts, vl_masks, config['patch_size'], stride=30, augment=False)
    tr_dl = DataLoader(tr_ds, config['batch_size'], shuffle=True, pin_memory=True)
    vl_dl = DataLoader(vl_ds, config['batch_size'], shuffle=False, pin_memory=True)

    # Part A: U-Net
    print("\n[Part A] 训练 U-Net")
    unet = UNet(3, 1, config['base_channels'])
    print(f"参数: {sum(p.numel() for p in unet.parameters()):,}")
    unet, hist_u = train_model(unet, tr_dl, vl_dl, "unet", config)
    plot_loss_curve(hist_u['train'], hist_u['val'], "U-Net Loss", f"{config['output_dir']}/unet_loss.png")
    _, avg_u, preds_u = evaluate_model(unet, test_imgs, test_gts, test_masks, config, "UNet")
    visualize_dl_result(test_imgs, test_gts, preds_u, "UNet", 5, config['output_dir'])
    ap = np.concatenate([p[1].flatten() for p in preds_u])
    ag = np.concatenate([g.flatten() for g in test_gts])
    am = np.concatenate([m.flatten() for m in test_masks])
    roc_u = compute_roc_metrics(ap, ag, am); avg_u['auc'] = roc_u['auc']
    plot_roc_curve(roc_u, "U-Net ROC", f"{config['output_dir']}/unet_roc.png")

    # Part B: Attention U-Net
    print("\n[Part B] 训练 Attention U-Net（扩展）")
    aunet = AttentionUNet(3, 1, config['base_channels'])
    print(f"参数: {sum(p.numel() for p in aunet.parameters()):,}")
    aunet, hist_a = train_model(aunet, tr_dl, vl_dl, "attention_unet", config)
    plot_loss_curve(hist_a['train'], hist_a['val'], "Attention U-Net Loss", f"{config['output_dir']}/attention_unet_loss.png")
    _, avg_a, preds_a = evaluate_model(aunet, test_imgs, test_gts, test_masks, config, "Attention UNet")
    visualize_dl_result(test_imgs, test_gts, preds_a, "Attention_UNet", 5, config['output_dir'])
    roc_a = compute_roc_metrics(np.concatenate([p[1].flatten() for p in preds_a]), ag, am)
    avg_a['auc'] = roc_a['auc']
    plot_roc_curve(roc_a, "Attention U-Net ROC", f"{config['output_dir']}/attention_unet_roc.png")

    # 对比
    print("\n方法对比:")
    print(f"{'Method':<25s} {'Acc':>8s} {'Se':>8s} {'Sp':>8s} {'F1':>8s} {'AUC':>8s}")
    print("-" * 70)
    for mn, mv in [("Unsupervised(Exp2)", {'accuracy':0.8745,'sensitivity':0.1707,'specificity':0.9772,'f1':0.2501,'auc':0.5740}),
                   ("U-Net", avg_u), ("Attention U-Net", avg_a)]:
        print(f"{mn:<25s} {mv['accuracy']:8.4f} {mv['sensitivity']:8.4f} {mv['specificity']:8.4f} {mv['f1']:8.4f} {mv['auc']:8.4f}")

    # ROC 对比图
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(roc_u['fpr'], roc_u['tpr'], label=f"U-Net (AUC={roc_u['auc']:.4f})")
    ax.plot(roc_a['fpr'], roc_a['tpr'], label=f"Attn U-Net (AUC={roc_a['auc']:.4f})")
    ax.plot([0,1],[0,1],'k--', label="Random"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Comparison"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{config['output_dir']}/roc_comparison.png", dpi=150, bbox_inches='tight'); plt.close()

    # 指标柱状对比
    fig, ax = plt.subplots(figsize=(10,6))
    mn_n = ['Accuracy','Sensitivity','Specificity','F1','AUC']; x = np.arange(5); w = 0.25
    vu = [avg_u.get(k,0) for k in ['accuracy','sensitivity','specificity','f1','auc']]
    va = [avg_a.get(k,0) for k in ['accuracy','sensitivity','specificity','f1','auc']]
    ax.bar(x-w/2, vu, w, label='U-Net', color='steelblue')
    ax.bar(x+w/2, va, w, label='Attention U-Net', color='coral')
    ax.set_xticks(x); ax.set_xticklabels(mn_n); ax.set_ylim(0,1.05)
    ax.set_title('DL Vessel Segmentation'); ax.legend(); ax.grid(axis='y', alpha=0.3)
    for i, (u, a) in enumerate(zip(vu, va)):
        ax.text(i-w/2, u+0.01, f'{u:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i+w/2, a+0.01, f'{a:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); plt.savefig(f"{config['output_dir']}/metrics_comparison.png", dpi=150, bbox_inches='tight'); plt.close()

    # 分割结果对比图
    for idx in range(min(3, len(test_imgs))):
        name, img = test_imgs[idx]; gt_b = (test_gts[idx]>127).astype(np.uint8)
        _, pu = preds_u[idx]; pb_u=(pu>0.5).astype(np.uint8)*255
        _, pa = preds_a[idx]; pb_a=(pa>0.5).astype(np.uint8)*255
        fig, axes = plt.subplots(2,4,figsize=(16,8))
        axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0,0].set_title("Original"); axes[0,0].axis('off')
        axes[0,1].imshow(test_gts[idx], cmap='gray'); axes[0,1].set_title("GT"); axes[0,1].axis('off')
        axes[0,2].imshow(pb_u, cmap='gray'); axes[0,2].set_title("U-Net"); axes[0,2].axis('off')
        axes[0,3].imshow(pb_a, cmap='gray'); axes[0,3].set_title("Attn U-Net"); axes[0,3].axis('off')
        def diff_im(pb, gt):
            d=np.zeros((*pb.shape,3),dtype=np.uint8)
            d[:,:,1]=(pb>127)&gt*255; d[:,:,2]=(pb>127)&(1-gt)*255; d[:,:,0]=((pb<=127)&gt)*255
            return d
        axes[1,0].imshow(diff_im(pb_u, gt_b)); axes[1,0].set_title("U-Net Diff"); axes[1,0].axis('off')
        axes[1,1].imshow(diff_im(pb_a, gt_b)); axes[1,1].set_title("Attn U-Net Diff"); axes[1,1].axis('off')
        ov_u=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).copy(); ov_u[pb_u>0]=[0,255,0]
        axes[1,2].imshow(ov_u); axes[1,2].set_title("U-Net Overlay"); axes[1,2].axis('off')
        ov_a=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).copy(); ov_a[pb_a>0]=[0,255,0]
        axes[1,3].imshow(ov_a); axes[1,3].set_title("Attn U-Net Overlay"); axes[1,3].axis('off')
        plt.suptitle(f"Comparison - {name}", fontsize=12); plt.tight_layout()
        plt.savefig(f"{config['output_dir']}/compare_{os.path.splitext(name)[0]}.png", dpi=150, bbox_inches='tight'); plt.close()

    # 保存详细指标
    with open(f"{config['output_dir']}/metrics.txt", 'w') as f:
        f.write("Method,Accuracy,Sensitivity,Specificity,F1,AUC\n")
        for mn, mv in [("Unsupervised", {'accuracy':0.8745,'sensitivity':0.1707,'specificity':0.9772,'f1':0.2501,'auc':0.5740}),
                       ("U-Net", avg_u), ("Attention U-Net", avg_a)]:
            f.write(f"{mn},{mv['accuracy']:.4f},{mv['sensitivity']:.4f},{mv['specificity']:.4f},{mv['f1']:.4f},{mv['auc']:.4f}\n")
    print(f"\n完成，结果保存在 {config['output_dir']}/")
    return avg_u, avg_a


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
