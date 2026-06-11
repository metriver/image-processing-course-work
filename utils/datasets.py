"""
数据集加载: BIPED (边缘检测) 和 DRIVE (眼底血管)
"""
import os
import numpy as np
import cv2
from glob import glob


# ---- BIPED ----

BIPED_INFO = """
BIPED 数据集
预期结构: datasets/BIPED/
  imgs/test/rgbr/*.jpg       测试图像
  imgs/train/rgbr/real/*.jpg 训练图像
  edge_maps/test/rgbr/*.png  测试标注
  edge_maps/train/rgbr/real/*.png 训练标注
"""


def check_biped(data_dir="datasets/BIPED"):
    d = os.path.join(data_dir, "imgs", "test", "rgbr")
    return os.path.isdir(d) and len(glob(os.path.join(d, "*.jpg"))) > 0


def check_mbiped(data_dir="datasets/MBIPED"):
    return check_biped(data_dir) or check_biped("datasets/BIPED")


def load_biped_test(data_dir="datasets/BIPED", max_images=None):
    """加载 BIPED 测试集，按文件名匹配图像和边缘标注"""
    img_dir = os.path.join(data_dir, "imgs", "test", "rgbr")
    edge_dir = os.path.join(data_dir, "edge_maps", "test", "rgbr")
    if not os.path.isdir(img_dir):
        print(f"BIPED 测试目录不存在: {img_dir}"); return [], []

    img_files = sorted(glob(os.path.join(img_dir, "*.jpg")))
    edge_files = sorted(glob(os.path.join(edge_dir, "*.png")))
    if max_images:
        img_files = img_files[:max_images]; edge_files = edge_files[:max_images]

    img_map = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
    edge_map = {os.path.splitext(os.path.basename(f))[0]: f for f in edge_files}
    common = sorted(set(img_map) & set(edge_map))

    images, edge_maps = [], []
    for name in common:
        img = cv2.imread(img_map[name], cv2.IMREAD_COLOR)
        edge = cv2.imread(edge_map[name], cv2.IMREAD_GRAYSCALE)
        if img is not None and edge is not None:
            images.append((os.path.basename(img_map[name]), img))
            edge_maps.append(edge)
    print(f"加载 {len(images)} 张 BIPED 测试图像")
    return images, edge_maps


def load_biped_train(data_dir="datasets/BIPED", max_images=None):
    """加载 BIPED 训练集"""
    img_dir = os.path.join(data_dir, "imgs", "train", "rgbr", "real")
    edge_dir = os.path.join(data_dir, "edge_maps", "train", "rgbr", "real")
    if not os.path.isdir(img_dir): return [], []

    img_files = sorted(glob(os.path.join(img_dir, "*.jpg")))
    edge_files = sorted(glob(os.path.join(edge_dir, "*.png")))
    if max_images: img_files = img_files[:max_images]

    img_map = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
    edge_map = {os.path.splitext(os.path.basename(f))[0]: f for f in edge_files}
    common = sorted(set(img_map) & set(edge_map))

    images, edge_maps = [], []
    for name in common:
        img = cv2.imread(img_map[name], cv2.IMREAD_COLOR)
        edge = cv2.imread(edge_map[name], cv2.IMREAD_GRAYSCALE)
        if img is not None and edge is not None:
            images.append((os.path.basename(img_map[name]), img))
            edge_maps.append(edge)
    print(f"加载 {len(images)} 张 BIPED 训练图像")
    return images, edge_maps


load_mbiped_test = load_biped_test


# ---- DRIVE ----

DRIVE_INFO = """
DRIVE 数据集
预期结构: datasets/DRIVE/
  images/  (01_test.tif ... 40_training.tif)
  manual/  (01_manual1.gif ... 40_manual1.gif)
  mask/    (01_mask.gif ... 40_mask.gif)
"""


def _read_image(filepath):
    """读图，OpenCV 不行就用 PIL（Windows 上 GIF 需要）"""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is not None: return img
    try:
        from PIL import Image
        return np.array(Image.open(filepath).convert('L'))
    except Exception as e:
        print(f"读取失败 {filepath}: {e}"); return None


def estimate_fov(image, threshold=10):
    """从眼底图估计 FOV mask"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def check_drive(data_dir="datasets/DRIVE"):
    """检查 DRIVE 是否存在"""
    img_d = os.path.join(data_dir, "images")
    man_d = os.path.join(data_dir, "manual")
    if os.path.isdir(img_d) and os.path.isdir(man_d):
        return len(glob(os.path.join(img_d, "*.tif"))) >= 20
    # 子目录结构
    for sub in ["test/images", "training/images"]:
        d = os.path.join(data_dir, sub)
        if os.path.isdir(d) and glob(os.path.join(d, "*")): return True
    return False


def load_drive_data(data_dir="datasets/DRIVE", train=True):
    """加载 DRIVE 数据。train=True→21-40, False→01-20"""
    flat_img = os.path.join(data_dir, "images")
    flat_man = os.path.join(data_dir, "manual")
    flat_msk = os.path.join(data_dir, "mask")
    is_flat = os.path.isdir(flat_img) and os.path.isdir(flat_man)

    start, end = (21, 41) if train else (1, 21)
    images, gts, masks = [], [], []

    for i in range(start, end):
        if is_flat:
            tag = "training" if train else "test"
            img_f = os.path.join(flat_img, f"{i:02d}_{tag}.tif")
            gt_f = os.path.join(flat_man, f"{i:02d}_manual1.gif")
            msk_f = os.path.join(flat_msk, f"{i:02d}_mask.gif")
        else:
            sub = "training" if train else "test"
            img_dir = os.path.join(data_dir, sub, "images")
            gt_dir = os.path.join(data_dir, sub, "1st_manual")
            msk_dir = os.path.join(data_dir, sub, "mask")
            imgs_l = sorted(glob(os.path.join(img_dir, "*")))
            gts_l = sorted(glob(os.path.join(gt_dir, "*")))
            msks_l = sorted(glob(os.path.join(msk_dir, "*")))
            idx = i - start
            if idx >= len(imgs_l): continue
            img_f = imgs_l[idx]
            gt_f = gts_l[idx] if idx < len(gts_l) else None
            msk_f = msks_l[idx] if idx < len(msks_l) else None

        if not os.path.exists(img_f): continue
        img = cv2.imread(img_f, cv2.IMREAD_COLOR)
        if img is None: continue

        gt = _read_image(gt_f) if gt_f and os.path.exists(gt_f) else None
        if gt is None: continue

        msk = _read_image(msk_f) if msk_f and os.path.exists(msk_f) else None
        if msk is None: msk = estimate_fov(img)

        images.append((os.path.basename(img_f), img))
        gts.append((gt > 128).astype(np.uint8) * 255)
        masks.append((msk > 128).astype(np.uint8))

    print(f"加载 {len(images)} 张 DRIVE {'训练' if train else '测试'}图像")
    return images, gts, masks


def auto_download_drive(data_dir="datasets/DRIVE"):
    if check_drive(data_dir): print("DRIVE 就绪"); return True
    print(DRIVE_INFO); return False


def auto_download_mbiped(data_dir="datasets/BIPED"):
    if check_biped(data_dir): print("BIPED 就绪"); return True
    print(BIPED_INFO); return False


if __name__ == "__main__":
    print("BIPED:"); check_biped()
    imgs, _ = load_biped_test(max_images=3)
    if imgs: print(f"  {imgs[0][0]} shape={imgs[0][1].shape}")
    print("DRIVE:"); check_drive()
    imgs, gts, mks = load_drive_data(train=False)
    if imgs: print(f"  {imgs[0][0]} shape={imgs[0][1].shape}")
