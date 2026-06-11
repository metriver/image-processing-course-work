"""
实验二：无监督方法眼底血管分割
流程：绿色通道 → CLAHE → 高斯+中值滤波 → 顶帽变换 → Otsu → 形态学后处理
"""
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.datasets import check_drive, load_drive_data, auto_download_drive, DRIVE_INFO
from utils.metrics import compute_binary_metrics, print_metrics_table, plot_roc_curve, compute_roc_metrics


def extract_green_channel(image):
    """眼底图像绿色通道——血管对比度最高"""
    if len(image.shape) == 3:
        return image[:, :, 1]
    return image


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE 局部直方图均衡，增强血管对比度"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def remove_small_regions(binary, min_area=30):
    """去掉面积小于 min_area 的连通分量"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 255
    return result


def morphological_cleanup(binary_image, kernel_size=3):
    """开运算去噪点 → 闭运算填空洞 → 面积滤波"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, small_k, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return remove_small_regions(closed, min_area=30)


def segment_vessels_unsupervised(image, params=None):
    """完整的无监督血管分割流水线"""
    if params is None:
        params = {'clahe_clip': 2.0, 'clahe_grid': (8, 8),
                  'gaussian_blur': 3, 'median_blur': 5, 'morph_kernel': 3}

    green = extract_green_channel(image)
    clahe_img = apply_clahe(green, params['clahe_clip'], params['clahe_grid'])
    gauss = cv2.GaussianBlur(clahe_img, (params['gaussian_blur'], params['gaussian_blur']), 0)
    median = cv2.medianBlur(gauss, params['median_blur'])
    # 顶帽变换突出亮血管区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(median, cv2.MORPH_TOPHAT, kernel)
    enhanced = cv2.add(median, tophat // 2)
    # Otsu 二值化
    otsu_thresh, otsu_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    vessel = morphological_cleanup(otsu_binary, params['morph_kernel'])
    inter = {'green_channel': green, 'clahe': clahe_img, 'gaussian': gauss,
             'median': median, 'tophat': tophat, 'enhanced': enhanced,
             'otsu_binary': otsu_binary, 'otsu_threshold': otsu_thresh}
    return vessel, inter


def optimize_parameters(train_images, train_gts, train_masks):
    """在训练集上网格搜索最优 CLAHE/Gaussian/Morph 参数"""
    print("参数优化（训练集网格搜索）...")
    param_grid = {'clahe_clip': [1.5, 2.0, 2.5, 3.0],
                  'gaussian_blur': [3, 5], 'morph_kernel': [2, 3, 4]}
    best_f1, best_params = 0, None
    n = min(10, len(train_images))
    total = len(param_grid['clahe_clip']) * len(param_grid['gaussian_blur']) * len(param_grid['morph_kernel'])
    with tqdm(total=total, desc="参数搜索") as pbar:
        for cc in param_grid['clahe_clip']:
            for gk in param_grid['gaussian_blur']:
                for mk in param_grid['morph_kernel']:
                    params = {'clahe_clip': cc, 'clahe_grid': (8, 8),
                              'gaussian_blur': gk, 'median_blur': 5, 'morph_kernel': mk}
                    fs = []
                    for i in range(n):
                        _, img = train_images[i]
                        pred, _ = segment_vessels_unsupervised(img, params)
                        fs.append(compute_binary_metrics(pred, train_gts[i], train_masks[i])['f1'])
                    avg_f = np.mean(fs)
                    if avg_f > best_f1: best_f1 = avg_f; best_params = params.copy()
                    pbar.update(1)
    print(f"最优: {best_params}, F1={best_f1:.4f}")
    return best_params, best_f1


def visualize_pipeline(image, intermediate, vessel_mask, gt_mask, name, out_dir="outputs/exp2"):
    """画出分割流水线各步骤"""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    steps = [
        ("Original", cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None),
        ("Green", intermediate['green_channel'], 'gray'),
        ("CLAHE", intermediate['clahe'], 'gray'),
        ("Gaussian", intermediate['gaussian'], 'gray'),
        ("Median", intermediate['median'], 'gray'),
        ("TopHat", intermediate['tophat'], 'gray'),
        ("Enhanced", intermediate['enhanced'], 'gray'),
        (f"Otsu(th={intermediate['otsu_threshold']:.0f})", intermediate['otsu_binary'], 'gray'),
        ("Morph", vessel_mask, 'gray'),
        ("Ground Truth", gt_mask, 'gray'),
    ]
    for i, (t, im, cm) in enumerate(steps):
        if i < 10:
            axes[i].imshow(im, cmap=cm) if cm else axes[i].imshow(im)
            axes[i].set_title(t, fontsize=9); axes[i].axis('off')
    for i in range(len(steps), 10): axes[i].axis('off')
    plt.suptitle(f"Pipeline - {name}", fontsize=14); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pipeline_{os.path.splitext(name)[0]}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_comparison(image, pred, gt, name, out_dir="outputs/exp2"):
    """预测 vs GT 对比（含差异图：绿=TP, 红=FP, 蓝=FN）"""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Original"); axes[0, 0].axis('off')
    axes[0, 1].imshow(gt, cmap='gray'); axes[0, 1].set_title("GT"); axes[0, 1].axis('off')
    axes[0, 2].imshow(pred, cmap='gray'); axes[0, 2].set_title("Prediction"); axes[0, 2].axis('off')
    gt_b = (gt > 127).astype(np.uint8); pb = (pred > 127).astype(np.uint8)
    diff = np.zeros((*pred.shape, 3), dtype=np.uint8)
    diff[:, :, 1] = (pb & gt_b) * 255
    diff[:, :, 2] = (pb & (1 - gt_b)) * 255
    diff[:, :, 0] = ((1 - pb) & gt_b) * 255
    axes[1, 0].imshow(diff); axes[1, 0].set_title("Diff (G=TP,R=FP,B=FN)"); axes[1, 0].axis('off')
    ov = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy(); ov[pb > 0] = [0, 255, 0]
    axes[1, 1].imshow(ov); axes[1, 1].set_title("Overlay (Green=Pred)"); axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    plt.suptitle(f"Comparison - {name}", fontsize=14); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"compare_{os.path.splitext(name)[0]}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("实验二：无监督方法眼底血管分割")
    data_dir = "datasets/DRIVE"
    if not check_drive(data_dir):
        print("DRIVE 数据集未找到！"); auto_download_drive(data_dir); return

    print("加载数据...")
    train_images, train_gts, train_masks = load_drive_data(data_dir, train=True)
    test_images, test_gts, test_masks = load_drive_data(data_dir, train=False)
    if len(train_images) == 0 or len(test_images) == 0:
        print("无法加载数据集"); return
    print(f"训练: {len(train_images)}, 测试: {len(test_images)}")

    best_params, best_f1 = optimize_parameters(train_images, train_gts, train_masks)
    if best_params is None:
        best_params = {'clahe_clip': 2.0, 'clahe_grid': (8, 8), 'gaussian_blur': 3,
                       'median_blur': 5, 'morph_kernel': 3}

    print("\n测试集评估...")
    predictions, metrics_list, all_probs = [], [], []
    for idx in tqdm(range(len(test_images)), desc="Testing"):
        name, img = test_images[idx]; gt = test_gts[idx]; mask = test_masks[idx]
        pred, inter = segment_vessels_unsupervised(img, best_params)
        predictions.append(pred)
        metrics_list.append(compute_binary_metrics(pred, gt, mask))
        all_probs.append(pred.astype(np.float64))
        if idx < 3:
            visualize_pipeline(img, inter, pred, gt, name, "outputs/exp2")
            visualize_comparison(img, pred, gt, name, "outputs/exp2")

    avg = {}
    for k in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'dice']:
        avg[k] = np.mean([m[k] for m in metrics_list])
    print_metrics_table(avg, "无监督血管分割 平均指标")

    # ROC
    ap = np.concatenate([p.flatten() for p in all_probs])
    ag = np.concatenate([g.flatten() for g in test_gts])
    am = np.concatenate([m.flatten() for m in test_masks])
    roc = compute_roc_metrics(ap, ag, am); avg['auc'] = roc['auc']
    plot_roc_curve(roc, "Unsupervised ROC", "outputs/exp2/roc_curve.png")

    # 汇总图
    n = len(predictions); cols = 5; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.atleast_1d(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])
    for i in range(n):
        name, img = test_images[i]
        pb = (predictions[i] > 127).astype(np.uint8)
        gt_b = (test_gts[i] > 127).astype(np.uint8)
        ov = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
        ov[pb > 0] = [0, 255, 0]; ov[gt_b > 0] = [255, 0, 0]
        axes[i].imshow(ov)
        axes[i].set_title(f"{name}\nF1={metrics_list[i]['f1']:.3f}", fontsize=7); axes[i].axis('off')
    for i in range(n, len(axes)): axes[i].axis('off')
    plt.suptitle("Unsupervised Vessel Segmentation (20 Test Images)", fontsize=14); plt.tight_layout()
    plt.savefig("outputs/exp2/all_test_results.png", dpi=200, bbox_inches='tight'); plt.close()

    # 指标柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    mn = ['Accuracy', 'Sensitivity', 'Specificity', 'F1', 'AUC']
    vs = [avg.get(k, 0) for k in ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']]
    bars = ax.bar(mn, vs, color=['steelblue', 'coral', 'seagreen', 'gold', 'purple'])
    for b, v in zip(bars, vs): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 1.0); ax.set_title('Exp2 Metrics'); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig("outputs/exp2/metrics_bar.png", dpi=150, bbox_inches='tight'); plt.close()

    # 保存每张指标
    os.makedirs("outputs/exp2", exist_ok=True)
    with open("outputs/exp2/metrics.txt", 'w') as f:
        f.write(f"Avg: Acc={avg['accuracy']:.4f} Se={avg['sensitivity']:.4f} "
                f"Sp={avg['specificity']:.4f} F1={avg['f1']:.4f} AUC={avg['auc']:.4f}\n\n")
        for (nm,_), m in zip(test_images, metrics_list):
            f.write(f"{nm}: Acc={m['accuracy']:.4f} Se={m['sensitivity']:.4f} "
                    f"Sp={m['specificity']:.4f} F1={m['f1']:.4f}\n")

    print("完成，结果保存在 outputs/exp2/")
    return avg, metrics_list


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
