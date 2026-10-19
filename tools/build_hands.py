#!/usr/bin/env python3
"""Build the moving-hands assets for the bundled mannequin photo (drivermover --hands).

Writes src/drivermover/examples/mannequin/hands/{bg.png, spokes.png, armR.png, armL.png, rig.json}:

  bg.png      clean plate of the arm region: both arms + hands and the wheel spokes removed (LaMa inpainting if
              `simple-lama-inpainting` is importable, else OpenCV Telea), hidden rim arcs re-painted from the
              visible rim (angular interpolation in the wheel plane + borrowed leather grain)
  spokes.png  RGBA: the spokes/hub as they look with the arms removed (alpha = spoke mask) - rotated at runtime
  armR.png    RGBA: right upper arm + forearm + hand cut from the photo (GrabCut-refined polygons)
  armL.png    RGBA: left forearm + hand (dim, partly outside the frame)
  rig.json    ROI, skeleton points (shoulder/elbow/wrist), wrist lines, home grip angles, ranges

Only needed when changing the photo/masks; the runtime needs nothing beyond numpy/opencv/pillow.
  python tools/build_hands.py            (uses LaMa when available: pip install torch simple-lama-inpainting --no-deps)
"""
import json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drivermover import config as C                      # noqa: E402
from drivermover.wheel import WheelCompositor, ellipse_frame  # noqa: E402

EX = ROOT / "src/drivermover/examples/mannequin"
OUT = EX / "hands"
ROI = (560, 740, 1152, 1260)             # x0 y0 x1 y1 of everything the rig may touch (below the head zone)

POLYS = {   # hand-traced on the photo, refined with GrabCut
  "armR":  [[603,1000],[605,960],[615,945],[650,940],[690,932],[720,922],[745,905],[770,885],[790,876],[810,872],[825,876],
            [840,890],[855,905],[870,905],[885,915],[892,935],[885,950],[870,955],[865,965],[860,985],[848,1000],[850,1015],
            [845,1025],[830,1030],[800,1032],[770,1022],[750,1010],[735,1002],[715,1000],[700,1018],[680,1025],[640,1027],[615,1020]],
  "uarmR": [[603,812],[615,800],[637,788],[655,789],[672,800],[674,820],[672,945],[605,950],[602,900]],
  "tipR":  [[866,900],[886,895],[903,903],[903,950],[884,957],[868,940]],     # finger/thumb tips curled behind the rim
  "armL":  [[1045,897],[1060,885],[1090,872],[1105,858],[1120,848],[1152,845],[1152,928],[1130,925],[1115,912],[1100,898],
            [1080,897],[1060,902]],
}
RIG = {
  "roi": list(ROI),
  "right": {"shoulder": [638, 803], "elbow": [632, 984], "wrist": [718, 962], "wrist_line": [[724, 918], [712, 1003]],
            "elbow_line": [[600, 952], [676, 940]], "home": 300.0, "range": [258.0, 338.0], "elbow_follow": 0.25},
  "left":  {"elbow": [1003, 903], "wrist": [1098, 884], "wrist_line": [[1100, 850], [1095, 905]],
            "home": 92.0, "range": [76.0, 108.0], "elbow_follow": 0.0, "grip_r": 0.8,
            "regrip_min_speed": 60.0},
  "grip_r": 0.9,
  "rim_fill_arcs": [[262, 338], [50, 130]],
}


def poly_mask(pts, shape):
  m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 1); return m


def grabcut(img, pts, band=10, iters=6):
  H, W = img.shape[:2]
  m = poly_mask(pts, (H, W))
  gc = np.full((H, W), cv2.GC_BGD, np.uint8)
  k = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
  dil, ero = cv2.dilate(m, k), cv2.erode(m, k)
  gc[dil > 0] = cv2.GC_PR_BGD; gc[m > 0] = cv2.GC_PR_FGD; gc[ero > 0] = cv2.GC_FGD
  ys, xs = np.nonzero(dil); x0, x1, y0, y1 = max(xs.min() - 5, 0), min(xs.max() + 5, W), max(ys.min() - 5, 0), min(ys.max() + 5, H)
  sub = gc[y0:y1, x0:x1].copy()
  cv2.grabCut(img[y0:y1, x0:x1], sub, None, np.zeros((1, 65)), np.zeros((1, 65)), iters, cv2.GC_INIT_WITH_MASK)
  out = np.zeros((H, W), np.uint8); out[y0:y1, x0:x1] = (sub == cv2.GC_FGD) | (sub == cv2.GC_PR_FGD)
  out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
  n, lab, st, _ = cv2.connectedComponentsWithStats(out)
  if n > 1: out = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
  ff = out.copy(); cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 1)
  return out | (1 - ff)


def masks(bgr):
  H, W = bgr.shape[:2]
  g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
  armR = grabcut(bgr, POLYS["armR"])
  tip = (poly_mask(POLYS["tipR"], (H, W)) & (g > 9)).astype(np.uint8)
  tip = cv2.morphologyEx(cv2.morphologyEx(tip, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
  u = grabcut(bgr, POLYS["uarmR"], band=4) & cv2.dilate(poly_mask(POLYS["uarmR"], (H, W)), np.ones((7, 7), np.uint8))
  armL = grabcut(bgr, POLYS["armL"], band=6)
  return (armR | tip | u).astype(np.uint8), armL.astype(np.uint8)


def inpaint(rgb, mask, dil=9):
  m = cv2.dilate(mask.astype(np.uint8), np.ones((dil, dil), np.uint8))
  X0, Y0, X1, Y1 = ROI[0] - 40, ROI[1] - 60, ROI[2], ROI[3]
  try:
    from simple_lama_inpainting import SimpleLama
    global _LAMA
    if "_LAMA" not in globals(): _LAMA = SimpleLama()
    out = np.asarray(_LAMA(Image.fromarray(rgb[Y0:Y1, X0:X1]), Image.fromarray(m[Y0:Y1, X0:X1] * 255)))[:Y1 - Y0, :X1 - X0]
    how = "LaMa"
  except ImportError:
    out = cv2.inpaint(rgb[Y0:Y1, X0:X1], m[Y0:Y1, X0:X1], 7, cv2.INPAINT_TELEA); how = "OpenCV Telea"
  full = rgb.copy(); full[Y0:Y1, X0:X1] = out
  return np.where(m[..., None] > 0, full, rgb), how


def rim_fill(img, src, c, A, hole, arcs, r0=0.78, r1=1.0):
  """Re-paint the rim band inside `arcs` (deg) from the visible rim: per radius, interpolate colour along the clock
  angle across the hidden arc, then add high-pass leather grain borrowed from a visible arc."""
  H, W = img.shape[:2]
  qs = np.arange(0, 360, 0.25); rs = np.linspace(r0 - 0.04, r1 + 0.03, 64)
  Q, R = np.meshgrid(np.radians(qs), rs, indexing="ij")
  X = (c[0] + R * (A[0, 0] * np.sin(Q) - A[0, 1] * np.cos(Q))).astype(np.float32)
  Y = (c[1] + R * (A[1, 0] * np.sin(Q) - A[1, 1] * np.cos(Q))).astype(np.float32)
  Pm = cv2.remap(src.astype(np.float32), X, Y, cv2.INTER_LINEAR)
  bad = (cv2.remap(hole.astype(np.float32), X, Y, cv2.INTER_LINEAR) > 0.01) | (X < 1) | (X > W - 2) | (Y < 1) | (Y > H - 2)
  inarc = np.zeros(len(qs), bool)
  for a, b in arcs: inarc |= (qs >= a) & (qs <= b)
  out = Pm.copy(); idx = np.arange(len(qs))
  for j in range(len(rs)):
    hl = inarc | bad[:, j]; good = ~hl
    if good.sum() < 10: continue
    for ch in range(3):
      out[hl, j, ch] = np.interp(idx[hl], idx[good], Pm[good, j, ch], period=len(qs))
  det = Pm - cv2.blur(Pm, (1, 49))
  for (a, b), sh in zip(arcs, (-75, 90)):
    rows = np.where((qs >= a) & (qs <= b))[0]; srows = (rows + int(sh / 0.25)) % len(qs)
    out[rows] += det[srows] * (~bad[srows])[..., None]
  x0, y0, x1, y1 = ROI
  ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
  Ai = np.linalg.inv(A); u = Ai[0, 0] * (xs - c[0]) + Ai[0, 1] * (ys - c[1]); v = Ai[1, 0] * (xs - c[0]) + Ai[1, 1] * (ys - c[1])
  r = np.hypot(u, v); q = np.degrees(np.arctan2(u, -v)) % 360
  samp = cv2.remap(out, ((r - rs[0]) / (rs[1] - rs[0])).astype(np.float32), (q / 0.25).astype(np.float32), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_WRAP)
  arcw = cv2.GaussianBlur(np.interp(q, qs, inarc.astype(float), period=360).astype(np.float32), (0, 0), 6)
  band = np.clip((r - r0) / 0.02, 0, 1) * np.clip(((r1 + 0.005) - r) / 0.015, 0, 1)
  hid = cv2.GaussianBlur(cv2.dilate(hole.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 2.0)[y0:y1, x0:x1]
  a = (band * np.clip(arcw * 1.5, 0, 1) * hid)[..., None]          # only where the rim was hidden by an arm
  res = img.astype(np.float32).copy()
  res[y0:y1, x0:x1] = res[y0:y1, x0:x1] * (1 - a) + samp * a
  return np.clip(res + 0.5, 0, 255).astype(np.uint8)


def main():
  photo = EX / "photo.jpg"
  rgb = np.asarray(Image.open(photo).convert("RGB")); bgr = rgb[..., ::-1].copy()
  cfg = C.load(None, photo)
  c, A = ellipse_frame(cfg["wheel"]["ellipse"], cfg["wheel"]["up_dir"])
  armR, armL = masks(bgr)
  wc = WheelCompositor(rgb, cfg, warp_spokes=True)
  X0, Y0, X1, Y1 = wc.roi
  sp = np.zeros(armR.shape, np.float32); sp[Y0:Y1, X0:X1] = wc._warp(wc.alpha, 0.0)
  spk = (sp > 0.05).astype(np.uint8)
  arms = armR | armL
  hole = cv2.dilate(arms, np.ones((11, 11), np.uint8))
  plate_a, how = inpaint(rgb, arms)
  plate_b, _ = inpaint(rgb, arms | spk)
  plate_a = rim_fill(plate_a, rgb, c, A, hole, RIG["rim_fill_arcs"])
  plate_b = rim_fill(plate_b, rgb, c, A, hole, RIG["rim_fill_arcs"])
  x0, y0, x1, y1 = ROI
  OUT.mkdir(parents=True, exist_ok=True)
  crop = lambda a: a[y0:y1, x0:x1]
  Image.fromarray(crop(plate_b)).save(OUT / "bg.png", optimize=True)
  # the left forearm is seen through the wheel there, so those pixels are not spoke
  spa = cv2.GaussianBlur(sp * (1 - cv2.dilate(armL, np.ones((7, 7), np.uint8))), (0, 0), 0.8)
  spk8 = np.clip(spa * 255 + 0.5, 0, 255).astype(np.uint8)
  Image.fromarray(np.dstack([crop(plate_a * cv2.dilate((spk8 > 0).astype(np.uint8), np.ones((5, 5), np.uint8))[..., None]), crop(spk8)])).save(
    OUT / "spokes.png", optimize=True)
  for name, m in (("armR", armR), ("armL", armL)):
    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.7)
    a = np.where(m > 0, np.maximum(a, 0.5), a)      # keep the cut tight; soft 1-2 px edge
    keep = cv2.dilate(m, np.ones((5, 5), np.uint8))[..., None]       # colour only near the arm: small files
    Image.fromarray(np.dstack([crop(rgb * keep), crop(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))])).save(OUT / f"{name}.png", optimize=True)
  rig = dict(RIG, plate=how, source="photo.jpg")
  (OUT / "rig.json").write_text(json.dumps(rig, indent=1) + "\n")
  print(f"wrote {OUT} (plate: {how}; arms {int(armR.sum())}+{int(armL.sum())} px, spokes {int(spk.sum())} px)")


if __name__ == "__main__":
  main()
