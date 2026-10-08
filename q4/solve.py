import json
import time
from pathlib import Path
from typing import Any as Q4Any
from typing import cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic

q4_started = time.perf_counter()

DATA = Path("data/q4")
OUT = Path("outputs/q4")
OUT.mkdir(parents=True, exist_ok=True)
WORK_SIZE = 900
BG_MARGIN = 0.3
PRIOR_THR = 0.55
RETINEX_SIGMA = 40


def texture_color_features(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    chans = [lab[..., i] for i in range(3)]
    for ksize, sigma in [(15, 3), (31, 6)]:
        for theta in np.arange(4) * np.pi / 4:
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, sigma * 2, 0.5, 0)
            kern -= kern.mean()
            resp = np.abs(cv2.filter2D(gray, cv2.CV_32F, kern))
            chans.append(cv2.GaussianBlur(resp, (0, 0), sigma))
    lbp = local_binary_pattern((gray * 255).astype(np.uint8), 8, 1.5, "uniform")
    return np.stack(chans, -1), lbp


def superpixel_descriptors(img, n_segments=2500):
    feats, lbp = texture_color_features(img)
    sp = slic(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        n_segments=n_segments,
        compactness=10,
        start_label=0,
    )
    n = sp.max() + 1
    flat = sp.ravel()
    count = np.bincount(flat, minlength=n).astype(float)

    def mean(values):
        return np.bincount(flat, values.ravel(), n) / count

    desc = [mean(feats[..., c]) for c in range(feats.shape[-1])]
    desc += [
        np.sqrt(np.maximum(mean(feats[..., c] ** 2) - desc[c] ** 2, 0))
        for c in range(3)
    ]
    desc += [mean((lbp == b).astype(float)) for b in range(10)]
    desc = np.stack(desc, 1)
    desc = (desc - desc.mean(0)) / (desc.std(0) + 1e-6)
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    centers = np.c_[mean(xx), mean(yy)]
    return sp, desc, centers


def knn_dist(query, ref, k=3):
    d2 = ((query[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    return np.sqrt(np.sort(d2, 1)[:, :k].mean(1))


def retinex(img, sigma):
    f = img.astype(np.float32) + 1
    lum = cv2.GaussianBlur(f.mean(-1), (0, 0), sigma)[..., None]
    r = np.log(f) - np.log(lum)
    lo, hi = np.percentile(r, 1), np.percentile(r, 99)
    return (np.clip((r - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def segment(img, clicks):
    h, w = img.shape[:2]
    s = WORK_SIZE / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pts = np.round(np.asarray(clicks) * s).astype(int)
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    extent = max(x1 - x0, y1 - y0)

    sp, desc, centers = superpixel_descriptors(small)
    path = np.zeros(small.shape[:2], np.uint8)
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        cv2.line(path, tuple(map(int, a)), tuple(map(int, b)), 1, 3)
    fg_sp = np.unique(sp[path > 0])
    m = BG_MARGIN * extent
    cx, cy = centers[:, 0], centers[:, 1]
    bg_sp = np.nonzero((cx < x0 - m) | (cx > x1 + m) | (cy < y0 - m) | (cy > y1 + m))[0]
    d_fg, d_bg = knn_dist(desc, desc[fg_sp]), knn_dist(desc, desc[bg_sp])
    prior = (d_bg / (d_fg + d_bg))[sp]

    gc_mask = np.where(prior > PRIOR_THR, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc_mask[np.isin(sp, bg_sp)] = cv2.GC_BGD
    for p in pts:
        cv2.circle(gc_mask, tuple(map(int, p)), int(0.02 * extent) + 3, cv2.GC_FGD, -1)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cast(Q4Any, cv2).grabCut(
        retinex(small, RETINEX_SIGMA), gc_mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK
    )
    fg = np.isin(gc_mask, [cv2.GC_FGD, cv2.GC_PR_FGD])

    lab, _ = ndi.label(fg)
    keep = np.unique(lab[pts[:, 1], pts[:, 0]])
    fg = ndi.binary_fill_holes(np.isin(lab, keep[keep > 0]))
    fg = cv2.morphologyEx(
        fg.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    soft = cv2.resize(fg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    soft = cv2.GaussianBlur(soft, (0, 0), 1.0 / s)
    full = (soft > 0.5).astype(np.uint8)
    for x, y in np.asarray(clicks):
        assert full[y, x], "click not in mask"
    return full, cv2.resize(prior.astype(np.float32), (w, h))


clicks_all = json.loads((DATA / "clicks.json").read_text())
results = []
for fname, objects in clicks_all.items():
    img = cast(np.ndarray, cv2.imread(str(DATA / fname)))
    for obj, info in objects.items():
        mask, prior = segment(img, info["clicks"])
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[..., 3] = mask * 255
        ys, xs = np.nonzero(mask)
        cv2.imwrite(str(OUT / f"{obj}.png"), rgba)
        cv2.imwrite(
            str(OUT / f"{obj}_cropped.png"),
            rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1],
        )
        cv2.imwrite(str(OUT / f"{obj}_mask.png"), mask * 255)
        print(
            f"{obj:15s} {fname}: area {mask.sum() / mask.size:.1%} of image, bbox "
            f"x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}], "
            f"all {len(info['clicks'])} clicks inside mask"
        )
        results.append((obj, img, info["clicks"], prior, mask, rgba))

fig, ax = plt.subplots(len(results), 4, figsize=(20, 4.6 * len(results)))
for row, (obj, img, clicks, prior, mask, rgba) in enumerate(results):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax[row, 0].imshow(rgb)
    ax[row, 0].scatter(*np.asarray(clicks).T, c="yellow", s=40, edgecolors="k")
    ax[row, 0].set_title(f"{obj}: input + clicks")
    ax[row, 1].imshow(prior, cmap="magma")
    ax[row, 1].set_title("superpixel texture/colour prior")
    over = rgb.copy()
    over[mask == 0] = (over[mask == 0] * 0.25).astype(np.uint8)
    ax[row, 2].imshow(over)
    ax[row, 2].set_title("final mask overlay")
    checker = (np.indices(mask.shape).sum(0) // 40 % 2)[..., None] * 60 + 170
    comp = rgba[..., :3][..., ::-1] * (mask[..., None]) + checker * (
        1 - mask[..., None]
    )
    ax[row, 3].imshow(comp.astype(np.uint8))
    ax[row, 3].set_title("RGBA cut-out (checkerboard = transparent)")
    for a in ax[row]:
        a.axis("off")
plt.tight_layout()
plt.savefig(OUT / "summary.jpg", dpi=70)

print(f"Q4 solve runtime: {time.perf_counter() - q4_started:.2f} s")
