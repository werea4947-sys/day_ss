import os
import numpy as np
import rasterio
from scipy.optimize import nnls
from sklearn.decomposition import PCA

# ----------------- 参数区 -----------------
DATA_DIR = r"C:\Users\29717\Desktop\LE07_L2SP_129039_20210112_20210207_02_T1"
PREFIX   = "LE07_L2SP_129039_20210112_20210207_02_T1"
# 裁剪窗口
row_off, col_off = 4000, 5000   # 起始行列
height, width    = 800, 800     # 裁剪大小
# -----------------------------------------

def sam_angle(a, b):
    """光谱角（弧度）"""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return np.arccos(np.clip(a @ b, -1, 1))

def load_clip():
    """读取并裁剪 6 个波段"""
    bands = []
    for b in [1,2,3,4,5,7]:
        path = os.path.join(DATA_DIR, f"{PREFIX}_SR_B{b}.TIF")
        with rasterio.open(path) as src:
            clip = src.read(1, window=rasterio.windows.Window(col_off, row_off, width, height))
            clip = clip.astype(np.float32)
            clip[clip <= 0] = np.nan
            bands.append(clip)
    cube = np.stack(bands, axis=-1)   # H,W,6
    mask = ~np.isnan(cube).any(axis=-1)
    return cube, mask

def ppi_endmembers(cube, mask, n_end=3, n_iter=1000):
    """PPI 提取端元"""
    Xorig = cube[mask]
    X = Xorig - Xorig.mean(axis=0)
    pca = PCA(n_components=min(5, X.shape[1]))
    Xpca = pca.fit_transform(X)
    scores = np.zeros(Xpca.shape[0], dtype=int)
    for _ in range(n_iter):
        vec = np.random.randn(Xpca.shape[1])
        vec /= np.linalg.norm(vec) + 1e-12
        proj = Xpca @ vec
        scores[np.argmax(proj)] += 1
        scores[np.argmin(proj)] += 1
    idx = np.argsort(scores)[-n_end:]
    return Xorig[idx]

def auto_label_by_angle(endmembers):
    """只用角度给端元贴标签"""
    labels = ["水体", "植被", "建筑"]
    # 先验参考光谱（Landsat 7 SR 经验值）
    ref = {
        "水体": np.array([0.06, 0.05, 0.05, 0.04, 0.03, 0.02]),
        "植被": np.array([0.08, 0.07, 0.06, 0.25, 0.15, 0.08]),
        "建筑": np.array([0.18, 0.20, 0.22, 0.21, 0.25, 0.18])
    }
    ref = {k: v/np.linalg.norm(v) for k, v in ref.items()}
    names = []
    for em in endmembers:
        angles = {k: sam_angle(em, v) for k, v in ref.items()}
        names.append(min(angles, key=angles.get))
    return names

def unmix_and_ratio(cube, mask, endmembers, labels):
    X = cube.reshape(-1, cube.shape[-1])
    E = np.array(endmembers).T
    abund = np.zeros((X.shape[0], E.shape[1]), dtype=np.float32)
    for i in range(X.shape[0]):
        if not np.isnan(X[i]).any():
            abund[i], _ = nnls(E, X[i])
    # 归一化到和为1（但允许<1）
    s = abund.sum(axis=1, keepdims=True) + 1e-12
    abund /= s

    # 合并到三类
    label_names = ["水体", "植被", "建筑"]
    label_map   = {lbl: i for i, lbl in enumerate(label_names)}
    n_class = len(label_names)
    class_abund = np.zeros((abund.shape[0], n_class))
    for i, lbl in enumerate(labels):
        if lbl in label_map:
            class_abund[:, label_map[lbl]] += abund[:, i]

    # 未解释部分
    total = class_abund.sum(axis=1)
    unexplained = 1.0 - total
    class_abund = np.hstack([class_abund, unexplained.reshape(-1, 1)])

    #  求平均（现在总和一定为1）
    ratio = class_abund.mean(axis=0)
    return ratio

# ----------------- 主流程 -----------------
cube, mask = load_clip()
print("裁剪区形状:", cube.shape)
endmembers = ppi_endmembers(cube, mask, n_end=3, n_iter=1000)
labels     = auto_label_by_angle(endmembers)
for em, lb in zip(endmembers, labels):
    print(f"端元 {lb}  均值反射率 {np.nanmean(em):.3f}")
ratio = unmix_and_ratio(cube, mask, endmembers, labels)
label_all = ["水体", "植被", "建筑", "未解释"]
for name, val in zip(label_all, ratio):
    print(f"{name}: {val*100:.2f}%")