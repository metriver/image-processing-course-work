"""
评估指标: Accuracy, Sensitivity, Specificity, Precision, F1, Dice, ROC/AUC
"""
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os


def compute_binary_metrics(pred, gt, mask=None):
    """计算二值分割指标，可选 FOV mask 限定评估区域"""
    p = (np.array(pred) > 127).astype(np.uint8)
    g = (np.array(gt) > 127).astype(np.uint8)
    if mask is not None:
        m = (np.array(mask) > 0).astype(np.uint8)
        p, g = p[m > 0], g[m > 0]

    tp = np.sum((p == 1) & (g == 1))
    tn = np.sum((p == 0) & (g == 0))
    fp = np.sum((p == 1) & (g == 0))
    fn = np.sum((p == 0) & (g == 1))
    total = tp + tn + fp + fn

    acc = (tp + tn) / total if total else 0
    se = tp / (tp + fn) if (tp + fn) else 0
    sp = tn / (tn + fp) if (tn + fp) else 0
    pr = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * pr * se / (pr + se) if (pr + se) else 0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0

    return {'accuracy': acc, 'sensitivity': se, 'specificity': sp,
            'precision': pr, 'f1': f1, 'dice': dice,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def compute_edge_metrics(pred_edge, gt_edge, tolerance=3):
    """带容差的边缘检测指标（膨胀 GT 以允许像素级偏差）"""
    p = (np.array(pred_edge) > 127).astype(np.uint8)
    g = (np.array(gt_edge) > 127).astype(np.uint8)

    if tolerance > 0:
        import cv2
        k = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
        g_d = cv2.dilate(g, k); p_d = cv2.dilate(p, k)
        tp = np.sum((p == 1) & (g_d == 1))
        fp = np.sum((p == 1) & (g_d == 0))
        fn = np.sum((p_d == 0) & (g == 1))
    else:
        tp = np.sum((p == 1) & (g == 1))
        fp = np.sum((p == 1) & (g == 0))
        fn = np.sum((p == 0) & (g == 1))

    pr = tp / (tp + fp) if (tp + fp) else 0
    rc = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
    return {'precision': pr, 'recall': rc, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def compute_roc_metrics(probs, gt, mask=None):
    """计算 FPR/TPR 和 AUC"""
    pf = np.array(probs, dtype=np.float64).flatten()
    gf = np.array(gt, dtype=np.uint8).flatten()
    if pf.max() > 1: pf /= 255.0
    gf = (gf > 127).astype(np.uint8)
    if mask is not None:
        mf = (np.array(mask).flatten() > 0).astype(np.uint8)
        pf, gf = pf[mf > 0], gf[mf > 0]
    fpr, tpr, _ = roc_curve(gf, pf)
    return {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}


def plot_roc_curve(roc_data, title="ROC Curve", save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(roc_data['fpr'], roc_data['tpr'], lw=2, label=f"AUC = {roc_data['auc']:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curve(train_losses, val_losses=None, title="Loss Curve", save_path=None):
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', lw=1.5, label='Train Loss')
    if val_losses and len(val_losses) > 0:
        plt.plot(epochs, val_losses, 'r-', lw=1.5, label='Val Loss')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_metrics_table(metrics, title="Metrics"):
    print(f"\n{'='*50}\n  {title}\n{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, (int, np.integer)): print(f"  {k:20s}: {v:>10d}")
        elif isinstance(v, (float, np.floating)): print(f"  {k:20s}: {v:>10.4f}")
    print(f"{'='*50}\n")
