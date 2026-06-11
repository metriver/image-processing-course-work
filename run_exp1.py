"""
运行实验一：BIPED 边缘检测，50张测试，输出可视化和指标
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.datasets import check_biped, load_biped_test
from utils.metrics import compute_edge_metrics
from exp1_edge_detection import (roberts_edge_detection, prewitt_edge_detection,
                                  sobel_edge_detection, log_edge_detection,
                                  canny_like_detection, visualize_canny_steps)

OUT = "outputs/exp1"; os.makedirs(OUT, exist_ok=True)

def main():
    if not check_biped("datasets/BIPED"): print("BIPED not found!"); return
    test_imgs, test_edges = load_biped_test("datasets/BIPED", 50)
    if not test_imgs: print("no images"); return
    print(f"{len(test_imgs)} test images")

    methods = {
        'Roberts': lambda i: roberts_edge_detection(i, 25)[0],
        'Prewitt': lambda i: prewitt_edge_detection(i, 40)[0],
        'Sobel': lambda i: sobel_edge_detection(i, 40)[0],
        'LoG': lambda i: log_edge_detection(i, 5, 1.0, 10)[0],
        'Canny-like': lambda i: canny_like_detection(i, 1.2, 0.04, 0.12)[0],
    }
    am = {n: {'p': [], 'r': [], 'f1': []} for n in methods}

    for idx in tqdm(range(len(test_imgs)), desc="评估"):
        _, img = test_imgs[idx]; gt = (test_edges[idx] > 127).astype(np.uint8) * 255
        for mn, mf in methods.items():
            m = compute_edge_metrics(mf(img), gt, tolerance=2)
            am[mn]['p'].append(m['precision']); am[mn]['r'].append(m['recall']); am[mn]['f1'].append(m['f1'])

    res = {}
    for mn in methods:
        p = np.mean(am[mn]['p']); r = np.mean(am[mn]['r']); f = np.mean(am[mn]['f1'])
        res[mn] = {'precision': p, 'recall': r, 'f1': f}
        print(f"  {mn:12s}: P={p:.4f}  R={r:.4f}  F1={f:.4f}")

    with open(f"{OUT}/metrics.txt", 'w') as fh:
        for mn in methods: fh.write(f"{mn}: P={res[mn]['precision']:.4f}, R={res[mn]['recall']:.4f}, F1={res[mn]['f1']:.4f}\n")

    # 前5张可视化
    for idx in range(min(5, len(test_imgs))):
        name, img = test_imgs[idx]; gt = (test_edges[idx] > 127).astype(np.uint8) * 255
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0,0].set_title("Original"); axes[0,0].axis('off')
        axes[1,0].imshow(gt, cmap='gray'); axes[1,0].set_title("GT"); axes[1,0].axis('off')
        for i, (mn, mf) in enumerate(methods.items()):
            if mn == 'Roberts': ed, mag = roberts_edge_detection(img, 25)
            elif mn == 'Prewitt': ed, mag = prewitt_edge_detection(img, 40)
            elif mn == 'Sobel': ed, mag, _ = sobel_edge_detection(img, 40)
            elif mn == 'LoG': ed, mag = log_edge_detection(img, 5, 1.0, 10)
            else: ed, inter = canny_like_detection(img); mag = inter['magnitude']
            axes[0,i+1].imshow(mag, cmap='hot'); axes[0,i+1].set_title(f"{mn}\nMag", fontsize=8); axes[0,i+1].axis('off')
            axes[1,i+1].imshow(ed, cmap='gray'); axes[1,i+1].set_title(f"{mn}\nEdges", fontsize=8); axes[1,i+1].axis('off')
        plt.suptitle(f"Edge Detection - {name}", fontsize=12); plt.tight_layout()
        plt.savefig(f"{OUT}/edges_{os.path.splitext(name)[0]}.png", dpi=150, bbox_inches='tight'); plt.close()

    visualize_canny_steps(test_imgs[0][1], OUT)

    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(methods.keys()); x = np.arange(len(names)); w = 0.25
    ax.bar(x-w, [res[n]['precision'] for n in names], w, label='Precision')
    ax.bar(x, [res[n]['recall'] for n in names], w, label='Recall')
    ax.bar(x+w, [res[n]['f1'] for n in names], w, label='F1')
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel('Score')
    ax.set_title('Edge Detection Performance'); ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/metrics_comparison.png", dpi=150, bbox_inches='tight'); plt.close()
    print(f"Done -> {OUT}/"); return res

if __name__ == "__main__": main()
