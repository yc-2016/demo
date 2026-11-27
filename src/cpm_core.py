"""
施工进度图像差分核心算法

流程：
    1. SIFT 提取特征 -> 暴力匹配 -> 取距离最优的前 N%
    2. RANSAC 估计单应矩阵，把"后期照片"透视变换到"前期照片"视角
    3. 直方图匹配做光照归一化（抑制整体明暗差异导致的假变化）
    4. 重叠区内做 SSIM 结构相似度 -> Otsu 分割 -> 连通域 -> 变化区域 bbox
    5. 可选：Canny 边缘游移检测，找"后期新增的结构线条"（楼板/柱/墙）
    6. 输出：对齐图、SSIM 热力图、差异标注图、变化网格图、result.json

设计取舍（实测结论）：
    - 光照归一化用直方图匹配，不要用 CLAHE（实测 CLAHE 会放大局部
      对比度差异，假变化反而更多）。
    - 全场景照片里地面/云/机械变动很大，差分噪声必然高。要得到有意义
      的"进度"信号，请用 --roi 把比对范围限制到建筑主体。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim


@dataclass
class Stats:
    before_size: str
    after_size: str
    ref_keypoints: int
    mov_keypoints: int
    raw_matches: int
    kept_matches: int
    ransac_inliers: int
    inlier_ratio: float
    overlap_ratio: float
    mean_ssim: float
    changed_pixel_ratio: float
    changed_regions: int
    changed_cells: int
    roi: list[int] | None
    new_edge_ratio: float | None


def _load(path: str, max_width: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读不到图片: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if max_width and img.shape[1] > max_width:
        s = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(img.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    return img


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def _align(ref_gray: np.ndarray, mov_gray: np.ndarray, nfeatures: int,
           keep_ratio: float, ransac_thresh: float):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp_r, des_r = sift.detectAndCompute(ref_gray, None)
    kp_m, des_m = sift.detectAndCompute(mov_gray, None)
    if des_r is None or des_m is None or len(kp_r) < 4 or len(kp_m) < 4:
        return None, None, len(kp_r or []), len(kp_m or []), 0, 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(bf.match(des_m, des_r), key=lambda m: m.distance)
    raw = len(matches)
    if raw < 4:
        return None, None, len(kp_r), len(kp_m), raw, 0, 0.0

    kept = matches[:max(4, int(raw * keep_ratio))]
    src = np.float32([kp_m[m.queryIdx].pt for m in kept]).reshape(-1, 1, 2)
    dst = np.float32([kp_r[m.trainIdx].pt for m in kept]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    n_inl = int(inl.sum()) if inl is not None else 0
    return H, inl, len(kp_r), len(kp_m), raw, len(kept), n_inl / len(kept)


def _wanchor(mov_shape, H, target_wh) -> np.ndarray:
    ones = np.ones(mov_shape[:2], np.uint8) * 255
    return cv2.warpPerspective(ones, H, target_wh, flags=cv2.INTER_NEAREST)


def _parse_roi(roi: str | None, w: int, h: int) -> tuple[int, int, int, int] | None:
    """--roi 用相对比例：x,y,w,h 都是 0~1。"""
    if not roi:
        return None
    vals = [float(v) for v in roi.split(",")]
    if len(vals) != 4:
        raise ValueError("--roi 需要 4 个用逗号分隔的比例值：x,y,w,h")
    x, y, rw, rh = vals
    return (int(x * w), int(y * h), int(rw * w), int(rh * h))


def _regions_from_mask(mask: np.ndarray, scores: np.ndarray,
                       min_area: float) -> list[dict]:
    n, labels, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(st[i, 0]), int(st[i, 1]), int(st[i, 2]),
                              int(st[i, 3]), int(st[i, 4]))
        if area < min_area:
            continue
        comp = labels[y:y + bh, x:x + bw] == i
        s = float(scores[y:y + bh, x:x + bw][comp].mean()) / 255.0
        out.append({"bbox": [x, y, bw, bh], "area_px": area,
                    "fill_ratio": round(area / float(bw * bh), 3),
                    "mean_ssim": round(s, 4)})
    out.sort(key=lambda r: -r["area_px"])
    return out


def compare_photos(before_path: str, after_path: str, out_dir: str,
                   max_width: int = 2000, nfeatures: int = 6000,
                   keep_ratio: float = 0.15, ransac_thresh: float = 3.0,
                   min_area_ratio: float = 0.0015, grid_cell: int = 96,
                   cell_thresh: float = 0.25, ssim_win: int = 0,
                   roi: str | None = None, detect_new_edges: bool = False,
                   tag: str = "") -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{tag}_" if tag else ""

    ref = _load(before_path, max_width)
    mov = _load(after_path, max_width)
    ref_g = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    mov_g = cv2.cvtColor(mov, cv2.COLOR_RGB2GRAY)

    H, inl, kp_r, kp_m, raw, kept, inlier_ratio = _align(
        ref_g, mov_g, nfeatures, keep_ratio, ransac_thresh)
    if H is None or not np.isfinite(H).all():
        raise RuntimeError("特征匹配失败：两图可能没有共视区域，或拍摄角度差异过大。")

    h, w = ref.shape[:2]
    aligned = cv2.warpPerspective(mov, H, (w, h))
    valid = _wanchor(mov.shape[:2], H, (w, h))
    overlap_ratio = float((valid > 0).sum()) / float(h * w)
    if overlap_ratio < 0.05:
        raise RuntimeError(
            f"重叠区域仅 {overlap_ratio:.1%}，对齐不可信，请确认两张是同一建筑同一视角。")

    aligned_g = cv2.warpPerspective(mov_g, H, (w, h))
    aligned_g = match_histograms(aligned_g, ref_g, channel_axis=None).astype(np.uint8)

    win = _odd(ssim_win or int(0.0017 * max(w, mov.shape[1]) + 13.7))
    a_blur = cv2.blur(ref_g, (win, win))
    b_blur = cv2.blur(aligned_g, (win, win))
    mean_ssim, full = ssim(a_blur, b_blur, full=True, data_range=255)
    ssim_img = (full * 255).astype(np.uint8)

    # ---- 计算区域：默认全图，可用 ROI 收窄 ----
    box = _parse_roi(roi, w, h)
    if box:
        rx, ry, rw2, rh2 = box
        area_mask = np.zeros((h, w), np.uint8)
        area_mask[ry:ry + rh2, rx:rx + rw2] = 255
    else:
        area_mask = np.full((h, w), 255, np.uint8)
    calc = cv2.bitwise_and(valid, area_mask)
    if (calc > 0).sum() < 1000:
        raise RuntimeError("ROI 内有效重叠像素过少，请调整 --roi。")

    valid_px = calc > 0
    scores = np.where(valid_px, ssim_img, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(scores[valid_px], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dissim = np.zeros_like(ssim_img, np.uint8)
    dissim[(scores < otsu) & valid_px] = 255
    dissim = cv2.morphologyEx(dissim, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    overlap_area = float(valid_px.sum())
    regions = _regions_from_mask(dissim, scores, max(120.0, overlap_area * min_area_ratio))
    mean_ssim_calc = float(np.mean(ssim_img[valid_px])) / 255.0
    changed_pixel_ratio = float((dissim > 0).sum()) / overlap_area

    # ---- 变化网格 ----
    grid = aligned.copy()
    cells = []
    for y in range(0, h, grid_cell):
        for x in range(0, w, grid_cell):
            y2, x2 = min(y + grid_cell, h), min(x + grid_cell, w)
            ov = calc[y:y2, x:x2] > 0
            if ov.sum() < (y2 - y) * (x2 - x) * 0.5:
                continue
            frac = float((dissim[y:y2, x:x2] > 0).sum()) / float(ov.sum())
            if frac >= cell_thresh:
                cells.append({"cell": [x, y, x2 - x, y2 - y],
                              "change_ratio": round(frac, 3)})
                alpha = min(0.65, 0.25 + frac * 0.6)
                blk = grid[y:y2, x:x2].astype(np.float32)
                grid[y:y2, x:x2] = (blk * (1 - alpha)
                                    + np.array([255, 40, 40], np.float32) * alpha).astype(np.uint8)
                cv2.rectangle(grid, (x, y), (x2, y2), (255, 90, 90), 1)
    cells.sort(key=lambda c: -c["change_ratio"])

    # ---- 可选：新增结构边缘 ----
    new_edge_ratio = None
    novel_img = None
    novel_regions: list[dict] = []
    if detect_new_edges:
        e1 = cv2.Canny(cv2.GaussianBlur(ref_g, (5, 5), 0), 60, 160)
        e2 = cv2.Canny(cv2.GaussianBlur(aligned_g, (5, 5), 0), 60, 160)
        e1d = cv2.dilate(e1, np.ones((7, 7), np.uint8))       # 容忍轻微错位
        novel = cv2.bitwise_and(e2, cv2.bitwise_not(e1d))
        novel = cv2.bitwise_and(novel, calc)
        novel = cv2.morphologyEx(novel, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        new_edge_ratio = float((novel > 0).sum()) / overlap_area
        novel_regions = _regions_from_mask(
            novel, np.full_like(ssim_img, 255), max(400.0, overlap_area * 0.002))
        novel_img = aligned.copy()
        for r in novel_regions:
            x, y, bw, bh = r["bbox"]
            cv2.rectangle(novel_img, (x, y), (x + bw, y + bh), (255, 140, 0), 3)

    # ---- 出图 ----
    outlined = aligned.copy()
    for r in regions:
        x, y, bw, bh = r["bbox"]
        cv2.rectangle(outlined, (x, y), (x + bw, y + bh), (255, 0, 0), 3)

    heat = cv2.cvtColor(cv2.applyColorMap(ssim_img, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    heat[calc == 0] = 0
    mask_rgb = cv2.cvtColor(calc, cv2.COLOR_GRAY2RGB)
    if box:
        for img_ in (outlined, grid, mask_rgb):
            cv2.rectangle(img_, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

    def save(name: str, im: np.ndarray) -> str:
        p = os.path.join(out_dir, f"{prefix}{name}.png")
        cv2.imwrite(p, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        return p

    paths = {
        "before": save("01_before", ref),
        "after_aligned": save("02_after_aligned", aligned),
        "side_by_side": save("03_side_by_side", np.hstack([ref, aligned])),
        "overlap_mask": save("04_overlap_mask", mask_rgb),
        "ssim_heatmap": save("05_ssim_heatmap", heat),
        "diff_outlined": save("06_diff_outlined", outlined),
        "change_grid": save("07_change_grid", grid),
        "change_mask": save("09_change_mask", cv2.cvtColor(dissim, cv2.COLOR_GRAY2RGB)),
    }
    if novel_img is not None:
        paths["new_structure"] = save("08_new_structure", novel_img)

    stats = Stats(
        before_size=f"{ref.shape[1]}x{ref.shape[0]}",
        after_size=f"{mov.shape[1]}x{mov.shape[0]}",
        ref_keypoints=kp_r, mov_keypoints=kp_m, raw_matches=raw,
        kept_matches=kept, ransac_inliers=int(inl.sum()) if inl is not None else 0,
        inlier_ratio=round(inlier_ratio, 4),
        overlap_ratio=round(overlap_ratio, 4),
        mean_ssim=round(mean_ssim_calc, 4),
        changed_pixel_ratio=round(changed_pixel_ratio, 4),
        changed_regions=len(regions), changed_cells=len(cells),
        roi=list(box) if box else None,
        new_edge_ratio=round(new_edge_ratio, 4) if new_edge_ratio is not None else None,
    )
    result = {
        "inputs": {"before": os.path.abspath(before_path),
                   "after": os.path.abspath(after_path)},
        "stats": asdict(stats),
        "otsu_threshold": float(otsu),
        "grid_cell": grid_cell,
        "regions": regions[:30],
        "top_changed_cells": cells[:20],
        "new_structure_regions": novel_regions[:20],
        "outputs": paths,
    }
    with open(os.path.join(out_dir, f"{prefix}result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
