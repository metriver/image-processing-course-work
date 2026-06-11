"""
实验一：图像边缘检测
实现 Roberts、Prewitt、Sobel、LoG 算子及类 Canny 流程，
使用 cv2.filter2D 进行卷积，不用现成的边缘检测函数。
"""
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.datasets import check_biped, load_biped_test, auto_download_mbiped, BIPED_INFO
from utils.metrics import compute_edge_metrics


def rgb_to_gray(image):
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def roberts_edge_detection(image, threshold=30):
    """Roberts 交叉算子，2x2 对角线差分"""
    gray = rgb_to_gray(image).astype(np.float64)
    kx = np.array([[1, 0], [0, -1]], dtype=np.float64)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    gx = cv2.filter2D(gray, cv2.CV_64F, kx)
    gy = cv2.filter2D(gray, cv2.CV_64F, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.clip(mag, 0, 255)
    edges = (mag > threshold).astype(np.uint8) * 255
    return edges, mag.astype(np.uint8)


def prewitt_edge_detection(image, threshold=50):
    """Prewitt 算子，3x3 邻域平均差分"""
    gray = rgb_to_gray(image).astype(np.float64)
    kx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    ky = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    gx = cv2.filter2D(gray, cv2.CV_64F, kx)
    gy = cv2.filter2D(gray, cv2.CV_64F, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.clip(mag, 0, 255)
    edges = (mag > threshold).astype(np.uint8) * 255
    return edges, mag.astype(np.uint8)


def sobel_edge_detection(image, threshold=50):
    """Sobel 算子，中心加权 3x3 差分"""
    gray = rgb_to_gray(image).astype(np.float64)
    kx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    ky = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    gx = cv2.filter2D(gray, cv2.CV_64F, kx)
    gy = cv2.filter2D(gray, cv2.CV_64F, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx) * 180.0 / np.pi
    mag = np.clip(mag, 0, 255)
    edges = (mag > threshold).astype(np.uint8) * 255
    return edges, mag.astype(np.uint8), direction


def detect_zero_crossings(image, threshold=15):
    """遍历 3x3 邻域，检测水平/垂直/对角方向的过零点"""
    h, w = image.shape
    edges = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            p = image[i-1:i+2, j-1:j+2]
            if p[1, 0] * p[1, 2] < 0 and abs(p[1, 0] - p[1, 2]) > threshold:
                edges[i, j] = 255
            elif p[0, 1] * p[2, 1] < 0 and abs(p[0, 1] - p[2, 1]) > threshold:
                edges[i, j] = 255
            elif p[0, 0] * p[2, 2] < 0 and abs(p[0, 0] - p[2, 2]) > threshold:
                edges[i, j] = 255
            elif p[0, 2] * p[2, 0] < 0 and abs(p[0, 2] - p[2, 0]) > threshold:
                edges[i, j] = 255
    return edges


def log_edge_detection(image, kernel_size=5, sigma=1.4, threshold=15):
    """LoG：先高斯平滑再拉普拉斯，检测过零点"""
    gray = rgb_to_gray(image).astype(np.float64)
    smoothed = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    lap_kernel = np.array([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]], dtype=np.float64)
    log_resp = cv2.filter2D(smoothed, cv2.CV_64F, lap_kernel)
    edges = detect_zero_crossings(log_resp, threshold)
    log_vis = np.clip(np.abs(log_resp), 0, 255).astype(np.uint8)
    return edges, log_vis


def non_maximum_suppression(magnitude, direction):
    """沿梯度方向保留局部最大值，细化边缘"""
    h, w = magnitude.shape
    nms = np.zeros((h, w), dtype=np.float64)
    direction = np.abs(direction) % 180
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            pixel = magnitude[i, j]
            angle = abs(direction[i, j])
            if (angle < 22.5) or (angle > 157.5):
                if pixel >= magnitude[i, j-1] and pixel >= magnitude[i, j+1]:
                    nms[i, j] = pixel
            elif 22.5 <= angle < 67.5:
                if pixel >= magnitude[i-1, j+1] and pixel >= magnitude[i+1, j-1]:
                    nms[i, j] = pixel
            elif 67.5 <= angle < 112.5:
                if pixel >= magnitude[i-1, j] and pixel >= magnitude[i+1, j]:
                    nms[i, j] = pixel
            else:
                if pixel >= magnitude[i-1, j-1] and pixel >= magnitude[i+1, j+1]:
                    nms[i, j] = pixel
    return nms


def double_threshold(image, low_ratio=0.05, high_ratio=0.15):
    """双阈值 + 8邻域连接：高阈值定强边，低阈值连弱边"""
    max_val = image.max()
    if max_val == 0:
        return np.zeros_like(image, dtype=np.uint8)
    high = max_val * high_ratio
    low = max_val * low_ratio
    strong = (image >= high).astype(np.uint8) * 255
    weak = ((image >= low) & (image < high)).astype(np.uint8) * 128
    edges = strong.copy()
    h, w = image.shape
    changed, it = True, 0
    while changed and it < 50:
        changed = False; it += 1
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if weak[i, j] == 128 and np.any(strong[i-1:i+2, j-1:j+2] == 255):
                    edges[i, j] = 255
                    strong[i, j] = 255
                    weak[i, j] = 0
                    changed = True
    return edges


def canny_like_detection(image, sigma=1.4, low_ratio=0.05, high_ratio=0.15):
    """类 Canny 流程：高斯平滑 → Sobel → NMS → 双阈值"""
    gray = rgb_to_gray(image).astype(np.float64)
    smoothed = cv2.GaussianBlur(gray, (5, 5), sigma)
    skx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    sky = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    gx = cv2.filter2D(smoothed, cv2.CV_64F, skx)
    gy = cv2.filter2D(smoothed, cv2.CV_64F, sky)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx) * 180.0 / np.pi
    nms = non_maximum_suppression(mag, direction)
    edges = double_threshold(nms, low_ratio, high_ratio)
    inter = {
        'smoothed': smoothed.astype(np.uint8),
        'gx': gx, 'gy': gy,
        'magnitude': np.clip(mag, 0, 255).astype(np.uint8),
        'nms': np.clip(nms, 0, 255).astype(np.uint8),
    }
    return edges, inter


def visualize_canny_steps(image, output_dir="outputs/exp1"):
    """画出 Canny 流程每一步的中间结果"""
    edges, inter = canny_like_detection(image)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    titles = ['Original', 'Gaussian', 'Gx', 'Gy', 'Magnitude', 'NMS+Threshold']
    imgs = [
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        inter['smoothed'], inter['gx'], inter['gy'],
        inter['magnitude'], edges,
    ]
    cmaps = [None, 'gray', 'RdBu', 'RdBu', 'hot', 'gray']
    for i, (t, im, cm) in enumerate(zip(titles, imgs, cmaps)):
        axes[i].imshow(im, cmap=cm) if cm else axes[i].imshow(im)
        axes[i].set_title(t, fontsize=12); axes[i].axis('off')
    plt.suptitle("Canny Pipeline", fontsize=14); plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "canny_pipeline.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Canny pipeline saved: {output_dir}/canny_pipeline.png")


def save_result_grid(name, img, roberts, prewitt, sobel, log, canny, output_dir):
    """保存一张图上所有方法的边缘结果"""
    base = os.path.splitext(name)[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Original"); axes[0, 0].axis('off')
    axes[0, 1].imshow(roberts, cmap='gray'); axes[0, 1].set_title("Roberts"); axes[0, 1].axis('off')
    axes[0, 2].imshow(prewitt, cmap='gray'); axes[0, 2].set_title("Prewitt"); axes[0, 2].axis('off')
    axes[0, 3].imshow(sobel, cmap='gray'); axes[0, 3].set_title("Sobel"); axes[0, 3].axis('off')
    axes[1, 0].imshow(log, cmap='gray'); axes[1, 0].set_title("LoG"); axes[1, 0].axis('off')
    axes[1, 1].imshow(canny, cmap='gray'); axes[1, 1].set_title("Canny-like"); axes[1, 1].axis('off')
    for i in range(2, 4): axes[1, i].axis('off')
    plt.suptitle(f"Edge Detection - {name}", fontsize=12); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"all_edges_{base}.png"), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("实验一：图像边缘检测")
    data_dir = "datasets/BIPED"

    if not check_biped(data_dir):
        print("BIPED 数据集未找到，尝试用 DRIVE 图像演示...")
        drive_dir = "datasets/DRIVE/images"
        if os.path.exists(drive_dir):
            img_files = sorted(glob(os.path.join(drive_dir, "*")))[:5]
            images = []
            for f in img_files:
                im = cv2.imread(f, cv2.IMREAD_COLOR)
                if im is not None: images.append((os.path.basename(f), im))
            if images:
                out_dir = "outputs/exp1"; os.makedirs(out_dir, exist_ok=True)
                for name, im in images:
                    ro, _ = roberts_edge_detection(im, 25)
                    pr, _ = prewitt_edge_detection(im, 40)
                    so, _, _ = sobel_edge_detection(im, 40)
                    lo, _ = log_edge_detection(im, threshold=10)
                    ca, _ = canny_like_detection(im)
                    save_result_grid(name, im, ro, pr, so, lo, ca, out_dir)
                visualize_canny_steps(images[0][1], out_dir)
                print(f"完成，结果保存在 {out_dir}/"); return
        print("未找到可用数据集。"); return

    test_images, test_edges = load_biped_test(data_dir, max_images=50)
    if len(test_images) == 0: print("无法加载测试图像"); return
    print(f"测试图像: {len(test_images)} 张")

    methods = {
        'Roberts':    lambda img: roberts_edge_detection(img, 25)[0],
        'Prewitt':    lambda img: prewitt_edge_detection(img, 40)[0],
        'Sobel':      lambda img: sobel_edge_detection(img, 40)[0],
        'LoG':        lambda img: log_edge_detection(img, 5, 1.0, 10)[0],
        'Canny-like': lambda img: canny_like_detection(img, 1.2, 0.04, 0.12)[0],
    }

    all_metrics = {n: {'precision': [], 'recall': [], 'f1': []} for n in methods}
    for idx in range(len(test_images)):
        name, img = test_images[idx]
        gt_bin = (test_edges[idx] > 127).astype(np.uint8) * 255
        for mname, mfn in methods.items():
            m = compute_edge_metrics(mfn(img), gt_bin, tolerance=2)
            all_metrics[mname]['precision'].append(m['precision'])
            all_metrics[mname]['recall'].append(m['recall'])
            all_metrics[mname]['f1'].append(m['f1'])

    print("\n各方法平均指标 (BIPED):")
    results = {}
    for mname in methods:
        p = np.mean(all_metrics[mname]['precision'])
        r = np.mean(all_metrics[mname]['recall'])
        f = np.mean(all_metrics[mname]['f1'])
        results[mname] = {'precision': p, 'recall': r, 'f1': f}
        print(f"  {mname:12s}: P={p:.4f}  R={r:.4f}  F1={f:.4f}")

    out_dir = "outputs/exp1"; os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/metrics.txt", 'w') as f:
        for n in methods:
            f.write(f"{n}: P={results[n]['precision']:.4f}, R={results[n]['recall']:.4f}, F1={results[n]['f1']:.4f}\n")

    # 前5张可视化
    for idx in range(min(5, len(test_images))):
        name, img = test_images[idx]
        gt_bin = (test_edges[idx] > 127).astype(np.uint8) * 255
        base = os.path.splitext(name)[0]
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Original"); axes[0, 0].axis('off')
        axes[1, 0].imshow(gt_bin, cmap='gray'); axes[1, 0].set_title("GT"); axes[1, 0].axis('off')
        for i, (mname, mfn) in enumerate(methods.items()):
            if mname == 'Roberts': ed, mag = roberts_edge_detection(img, 25)
            elif mname == 'Prewitt': ed, mag = prewitt_edge_detection(img, 40)
            elif mname == 'Sobel': ed, mag, _ = sobel_edge_detection(img, 40)
            elif mname == 'LoG': ed, mag = log_edge_detection(img, 5, 1.0, 10)
            else: ed, inter = canny_like_detection(img); mag = inter['magnitude']
            axes[0, i+1].imshow(mag, cmap='hot'); axes[0, i+1].set_title(f"{mname}\nMagnitude", fontsize=8); axes[0, i+1].axis('off')
            axes[1, i+1].imshow(ed, cmap='gray'); axes[1, i+1].set_title(f"{mname}\nEdges", fontsize=8); axes[1, i+1].axis('off')
        plt.suptitle(f"Edge Detection - {name}", fontsize=12); plt.tight_layout()
        plt.savefig(f"{out_dir}/edges_{base}.png", dpi=150, bbox_inches='tight'); plt.close()

    visualize_canny_steps(test_images[0][1], out_dir)

    # 柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(methods.keys()); x = np.arange(len(names)); w = 0.25
    ax.bar(x-w, [results[n]['precision'] for n in names], w, label='Precision')
    ax.bar(x, [results[n]['recall'] for n in names], w, label='Recall')
    ax.bar(x+w, [results[n]['f1'] for n in names], w, label='F1')
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('Score'); ax.set_title('Edge Detection Performance')
    ax.legend(); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(f"{out_dir}/metrics_comparison.png", dpi=150, bbox_inches='tight'); plt.close()

    print(f"\n结果保存在 {out_dir}/")
    return results


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
