"""
压力测试：人为制造视角差异，找出配准的失效边界。

背景：我们目前只在 TNB/PNLC 这类"机位几乎不动"的照片上验证过（重叠 99.8%）。
真实场景里机位会平移、俯仰、变焦，需要知道这套 SIFT + RANSAC 在多大差异下还撑得住。

做法：对"后期照片"施加可控的几何扰动（透视、缩放、平移），
逐级加大，看配准能否恢复、SSIM 是否稳定。

用法：python src/stress_test.py --before a.jpg --after b.jpg
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from skimage.exposure import match_histograms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpm_core import _align, _load, _wanchor  # noqa: E402


def warp_perspective(img: np.ndarray, strength: float, rng) -> np.ndarray:
    """模拟机位偏转：四角按比例随机位移（相当于相机稍微转向）。"""
    h, w = img.shape[:2]
    d = strength * w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src.copy()
    for i in range(4):
        dst[i, 0] += rng.uniform(-d, d)
        dst[i, 1] += rng.uniform(-d, d)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(0, 0, 0))


def warp_scale(img: np.ndarray, factor: float) -> np.ndarray:
    """模拟变焦：中心等比缩放后居中放置。"""
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * factor), int(h * factor)),
                       interpolation=cv2.INTER_AREA)
    canvas = np.zeros_like(img)
    y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return canvas


def warp_translate(img: np.ndarray, ratio: float) -> np.ndarray:
    """模拟机位横移：整体平移（保持朝向）。"""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, ratio * w], [0, 1, ratio * h * 0.3]])
    return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))


def evaluate(ref: np.ndarray, mov: np.ndarray) -> dict:
    """跑一次配准，返回指标。"""
    ref_g = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    mov_g = cv2.cvtColor(mov, cv2.COLOR_RGB2GRAY)
    try:
        H, inl, kp_r, kp_m, raw, kept, ratio = _align(ref_g, mov_g, 6000, 0.15, 3.0)
    except Exception as e:                                  # noqa: BLE001
        return {"ok": False, "reason": f"异常 {type(e).__name__}"}
    if H is None or not np.isfinite(H).all():
        return {"ok": False, "reason": "单应矩阵求解失败"}

    h, w = ref.shape[:2]
    overlap = _wanchor(mov.shape[:2], H, (w, h))
    ov_ratio = float((overlap > 0).sum()) / (h * w)
    if ov_ratio < 0.05:
        return {"ok": False, "reason": f"重叠仅 {ov_ratio:.0%}", "overlap": ov_ratio}

    ag = cv2.warpPerspective(mov_g, H, (w, h))
    ag = match_histograms(ag, ref_g, channel_axis=None).astype(np.uint8)
    from skimage.metrics import structural_similarity as ssim
    win = 21
    a = cv2.blur(ref_g, (win, win))
    b = cv2.blur(ag, (win, win))
    _, full = ssim(a, b, full=True, data_range=255)
    mean_ssim = float(np.mean((full * 255)[overlap > 0])) / 255.0

    return {"ok": True, "inliers": int(inl.sum()) if inl is not None else 0,
            "inlier_ratio": round(ratio, 3), "overlap": round(ov_ratio, 3),
            "ssim": round(mean_ssim, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--max-width", type=int, default=1600)
    args = ap.parse_args()

    ref = _load(args.before, args.max_width)
    mov = _load(args.after, args.max_width)
    rng = np.random.default_rng(42)

    base = evaluate(ref, mov)
    print("=" * 74)
    print(f" 基准（无扰动）：{'成功' if base['ok'] else base['reason']}  "
          f"内点率 {base.get('inlier_ratio')}  重叠 {base.get('overlap')}  "
          f"SSIM {base.get('ssim')}")
    print("=" * 74)
    print(f"{'扰动类型':10s}{'强度':>8s}{'结果':>8s}{'内点率':>9s}{'重叠':>8s}{'SSIM':>9s}{'SSIM降幅':>10s}")

    base_ssim = base.get("ssim", 0.0)
    tests = [("透视偏转", s, lambda im, s=s: warp_perspective(im, s, rng))
             for s in (0.02, 0.05, 0.10, 0.15, 0.25)]
    tests += [("变焦", f, lambda im, f=f: warp_scale(im, f))
              for f in (0.9, 0.8, 0.65, 0.5)]
    tests += [("平移", r, lambda im, r=r: warp_translate(im, r))
              for r in (0.05, 0.10, 0.20, 0.35)]

    for name, strength, fn in tests:
        r = evaluate(ref, fn(mov))
        if r["ok"]:
            drop = base_ssim - r["ssim"]
            print(f"{name:10s}{strength:>8.2f}{'成功':>8s}{r['inlier_ratio']:>9.3f}"
                  f"{r['overlap']:>8.3f}{r['ssim']:>9.4f}{drop:>+10.4f}")
        else:
            print(f"{name:10s}{strength:>8.2f}{'失败':>8s}  {r['reason']}")
    print()


if __name__ == "__main__":
    main()
