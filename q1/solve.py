import time
from pathlib import Path
from typing import cast

import cv2
import matplotlib.pyplot as plt
import numpy as np

q1_started = time.perf_counter()

DATA = Path("data/q1")
OUT = Path("outputs/q1")
OUT.mkdir(parents=True, exist_ok=True)


def grass_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    mask = ((h > 20) & (h < 90) & (s > 60)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def horizon_row(grass):
    rows = grass.mean(axis=1)
    return int(np.argmax(rows > 0.5))


def runway_mask(grass, y_h):
    cand = 1 - grass
    cand[: y_h + 5] = 0
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    _, lab, stats, _ = cv2.connectedComponentsWithStats(cand)
    best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (lab == best).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def side_points(mask):
    ys, left, right = [], [], []
    rows = np.where(mask.any(axis=1))[0]
    y0, y1 = rows.min(), rows.max()
    margin = int(0.04 * (y1 - y0))
    for y in range(y0 + margin, y1 - margin):
        xs = np.where(mask[y])[0]
        ys.append(y)
        left.append(xs.min())
        right.append(xs.max())
    ys = np.array(ys, float)
    return ys, np.array(left, float), np.array(right, float), (y0, y1)


def refine_with_edges(gray, ys, xs, band=6):
    gx = cv2.Sobel(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_32F, 1, 0, ksize=3)
    out = []
    for y, x in zip(ys.astype(int), xs.astype(int), strict=True):
        lo, hi = max(x - band, 0), min(x + band + 1, gray.shape[1])
        out.append(lo + np.argmax(np.abs(gx[y, lo:hi])))
    return np.array(out, float)


def robust_line(ys, xs):
    pts = np.c_[xs, ys].astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    a = vx / vy
    b = x0 - a * y0
    resid = np.abs(xs - (a * ys + b))
    return a, b, resid


img = cast(np.ndarray, cv2.imread(str(DATA / "Runway.jpg")))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grass = grass_mask(img)
y_h = horizon_row(grass)
runway = runway_mask(grass, y_h)
ys, xl, xr, (y_top, y_bot) = side_points(runway)
xl_ref = refine_with_edges(gray, ys, xl)
xr_ref = refine_with_edges(gray, ys, xr)

lines = {}
for name, xs in [("left", xl_ref), ("right", xr_ref)]:
    a, b, resid = robust_line(ys, xs)
    lines[name] = (a, b)
    print(
        f"{name:5s}: x = {a:+.4f}*y + {b:.1f}  points {len(ys)}  "
        f"mean |res| {resid.mean():.2f}px  max |res| {resid.max():.2f}px  "
        f"within 3px {np.mean(resid < 3):.1%}"
    )

(al, bl), (ar, br) = lines["left"], lines["right"]
y_vp = (br - bl) / (al - ar)
x_vp = al * y_vp + bl
print(f"horizon row (grass onset) y={y_h}, runway rows {y_top}..{y_bot}")
print(f"vanishing point of the two boundaries: ({x_vp:.1f}, {y_vp:.1f})")

endpoints = {}
for name, (a, b) in lines.items():
    p_top = (int(round(a * y_top + b)), int(y_top))
    p_bot = (int(round(a * y_bot + b)), int(y_bot))
    endpoints[name] = (p_top, p_bot)
    print(f"{name} boundary segment: {p_bot} -> {p_top}")

annot = img.copy()
for p_top, p_bot in endpoints.values():
    cv2.line(annot, p_bot, p_top, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imwrite(str(OUT / "runway_boundaries.png"), annot)
x0c = max(min(p[0] for p in sum(endpoints.values(), ())) - 40, 0)
x1c = max(p[0] for p in sum(endpoints.values(), ())) + 40
cv2.imwrite(
    str(OUT / "runway_boundaries_zoom.png"),
    annot[max(y_top - 20, 0) : y_bot + 20, x0c:x1c],
)

fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
ax[0].imshow(grass, cmap="gray")
ax[0].set_title("Grass mask (HSV)")
ax[1].imshow(runway, cmap="gray")
ax[1].set_title("Runway component")
ax[1].plot(xl_ref, ys, "r.", ms=1)
ax[1].plot(xr_ref, ys, "c.", ms=1)
ax[2].imshow(cv2.cvtColor(annot, cv2.COLOR_BGR2RGB))
ax[2].set_title("Detected boundaries")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(OUT / "pipeline.png", dpi=110)

print(f"Q1 solve runtime: {time.perf_counter() - q1_started:.2f} s")
