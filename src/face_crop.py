"""
把"施工作业面"裁出来放大，供视觉模型数楼层。

动机：远景照片里一层的楼板线只有 3~5 个像素，模型数不准（实测误差 ±2 层，
而两期真实增量才 1~3 层，信噪比不足）。裁剪后放大到 1600px，
同样的楼层线能到 10~15 像素，是可做的最直接的精度提升——不需要新数据。

做法：
1. 用 CV 差分得到的变化掩膜，取面积够大的连通域（就是正在施工的塔楼）
2. 求这些连通域的联合外接框，向外扩一圈留上下文
3. 在【前期】和【后期】同一坐标范围各裁一块，等比放大到目标边长
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def locate_face(mask: np.ndarray, min_ratio: float = 0.0008,
                pad: float = 0.10, top_zone: float = 0.28) -> tuple[int, int, int, int] | None:
    """
    在变化掩膜里找"作业面"（塔楼顶部正在施工的那一段），返回 (x, y, w, h)。

    关键：不能对所有连通域取联合外接框——画面底部的地面材料带、围挡、阴影
    面积最大，会把框拉成整幅图。真正的作业面在画面【上部】，所以只在
    y 小于 top_zone*H 的范围内找，并取面积最大的那一块（通常是主塔楼）。
    """
    if mask is None or mask.size == 0:
        return None
    h_img, w_img = mask.shape[:2]
    m = (mask > 127).astype(np.uint8) * 255
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    min_area = h_img * w_img * min_ratio

    def pick(zone: float):
        best = None
        for i in range(1, n):
            x, y, w, h, area = (int(stats[i, 0]), int(stats[i, 1]),
                                int(stats[i, 2]), int(stats[i, 3]),
                                int(stats[i, 4]))
            if area < min_area or w < w_img * 0.05:
                continue
            if y > h_img * zone:          # 只看画面上部
                continue
            if h < h_img * 0.05:          # 排除细长横条
                continue
            # 排除"又宽又扁"的地面带（材料堆、围挡、阴影连成的长条），
            # 它们的面积往往最大，会把框拉成整幅图
            if w > w_img * 0.8 and h / max(w, 1) < 0.5:
                continue
            if best is None or area > best[4]:
                best = (x, y, w, h, area)
        return best

    # 先找顶部作业面；工地在早期阶段（楼还没长高）时再放宽
    best = pick(top_zone) or pick(0.55)
    if best is None:
        return None

    x, y, w, h, _ = best
    dx = int(w * pad)
    up = int(h * 0.06)
    down = int(h * 0.42)                  # 往下多留几层已完工楼层作参照
    x0, y0 = max(0, x - dx), max(0, y - up)
    x1 = min(w_img, x + w + dx)
    y1 = min(h_img, y + h + down)
    return x0, y0, x1 - x0, y1 - y0


def crop_pair(before_img: np.ndarray, after_img: np.ndarray,
              box: tuple[int, int, int, int], target_edge: int = 1600):
    """按同一坐标范围裁剪，并等比放大到目标边长。"""
    x, y, w, h = box
    crops = []
    for img in (before_img, after_img):
        c = img[max(0, y):y + h, max(0, x):x + w]
        if c.size == 0:
            crops.append(img.copy())
            continue
        ch, cw = c.shape[:2]
        scale = target_edge / max(ch, cw)
        if scale > 1.0:
            c = cv2.resize(c, (int(cw * scale), int(ch * scale)),
                           interpolation=cv2.INTER_CUBIC)
        crops.append(c)
    return crops[0], crops[1]


def make_face_crops(before_rgb: np.ndarray, after_rgb: np.ndarray,
                    mask: np.ndarray, out_dir: str, prefix: str = "",
                    target_edge: int = 1600, pad: float = 0.10):
    """产出裁剪放大的一对图，返回 (前期路径, 后期路径, bbox)；找不到作业面返回 None。"""
    box = locate_face(mask, pad=pad)
    if box is None:
        return None
    b, a = crop_pair(before_rgb, after_rgb, box, target_edge)
    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, f"{prefix}10_face_before.png")
    p2 = os.path.join(out_dir, f"{prefix}11_face_after.png")
    cv2.imwrite(p1, cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
    cv2.imwrite(p2, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    return p1, p2, box
