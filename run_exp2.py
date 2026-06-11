"""
运行实验二：DRIVE 无监督血管分割，网格搜索参数，输出20张测试结果
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.datasets import check_drive, load_drive_data
from utils.metrics import compute_binary_metrics, print_metrics_table, plot_roc_curve, compute_roc_metrics
from exp2_unsupervised_vessel import segment_vessels_unsupervised, visualize_pipeline, visualize_comparison

OUT = "outputs/exp2"; os.makedirs(OUT, exist_ok=True)

def main():
    if not check_drive("datasets/DRIVE"): print("DRIVE not found!"); return
    tr_imgs, tr_gts, tr_masks = load_drive_data("datasets/DRIVE", train=True)
    te_imgs, te_gts, te_masks = load_drive_data("datasets/DRIVE", train=False)
    print(f"Train:{len(tr_imgs)} Test:{len(te_imgs)}")

    # 网格搜索
    pg = {'clahe_clip':[1.5,2.0,2.5,3.0], 'gaussian_blur':[3,5], 'morph_kernel':[2,3,4]}
    bf, bp = 0, {'clahe_clip':2.0,'clahe_grid':(8,8),'gaussian_blur':3,'median_blur':5,'morph_kernel':3}
    n = min(10, len(tr_imgs))
    for cc in pg['clahe_clip']:
        for gk in pg['gaussian_blur']:
            for mk in pg['morph_kernel']:
                p = {'clahe_clip':cc,'clahe_grid':(8,8),'gaussian_blur':gk,'median_blur':5,'morph_kernel':mk}
                fs = []
                for i in range(n):
                    _, img = tr_imgs[i]; pred, _ = segment_vessels_unsupervised(img, p)
                    fs.append(compute_binary_metrics(pred, tr_gts[i], tr_masks[i])['f1'])
                if np.mean(fs) > bf: bf = np.mean(fs); bp = p.copy()
    print(f"Best params: {bp} F1={bf:.4f}")

    # 测试
    preds, mlist, probs = [], [], []
    for idx in tqdm(range(len(te_imgs)), desc="Testing"):
        nm, img = te_imgs[idx]; gt = te_gts[idx]; mk = te_masks[idx]
        pred, inter = segment_vessels_unsupervised(img, bp)
        preds.append(pred); mlist.append(compute_binary_metrics(pred, gt, mk)); probs.append(pred.astype(np.float64))
        if idx < 3:
            visualize_pipeline(img, inter, pred, gt, nm, OUT); visualize_comparison(img, pred, gt, nm, OUT)

    avg = {}
    for k in ['accuracy','sensitivity','specificity','precision','f1','dice']:
        avg[k] = np.mean([m[k] for m in mlist])
    roc = compute_roc_metrics(np.concatenate([p.flatten() for p in probs]),
                               np.concatenate([g.flatten() for g in te_gts]),
                               np.concatenate([m.flatten() for m in te_masks]))
    avg['auc'] = roc['auc']
    print_metrics_table(avg, "无监督血管分割")
    plot_roc_curve(roc, "Unsupervised ROC", f"{OUT}/roc_curve.png")

    # 汇总图
    n = len(preds); cols = 5; rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.atleast_1d(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])
    for i in range(n):
        nm, img = te_imgs[i]; pb = (preds[i]>127).astype(np.uint8); gt_b = (te_gts[i]>127).astype(np.uint8)
        ov = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy(); ov[pb>0]=[0,255,0]; ov[gt_b>0]=[255,0,0]
        axes[i].imshow(ov); axes[i].set_title(f"{nm}\nF1={mlist[i]['f1']:.3f}", fontsize=7); axes[i].axis('off')
    for i in range(n, len(axes)): axes[i].axis('off')
    plt.suptitle("Unsupervised Vessel Segmentation (20 Test Images)", fontsize=14); plt.tight_layout()
    plt.savefig(f"{OUT}/all_test_results.png", dpi=200, bbox_inches='tight'); plt.close()

    # 柱状图
    fig, ax = plt.subplots(figsize=(8,5))
    vs = [avg.get(k,0) for k in ['accuracy','sensitivity','specificity','f1','auc']]
    bars = ax.bar(['Accuracy','Sensitivity','Specificity','F1','AUC'], vs,
                  color=['steelblue','coral','seagreen','gold','purple'])
    for b,v in zip(bars,vs): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0,1.0); ax.set_title('Exp2 Metrics'); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/metrics_bar.png", dpi=150, bbox_inches='tight'); plt.close()

    with open(f"{OUT}/metrics.txt", 'w') as f:
        f.write(f"Avg: Acc={avg['accuracy']:.4f} Se={avg['sensitivity']:.4f} Sp={avg['specificity']:.4f} F1={avg['f1']:.4f} AUC={avg['auc']:.4f}\n\n")
        for (nm,_), m in zip(te_imgs, mlist):
            f.write(f"{nm}: Acc={m['accuracy']:.4f} Se={m['sensitivity']:.4f} Sp={m['specificity']:.4f} F1={m['f1']:.4f}\n")
    print(f"Done -> {OUT}/"); return avg, mlist

if __name__ == "__main__": main()
