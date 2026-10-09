"""Realistic-head mode: composite head images (generated keyframes) onto the photo, chosen by the logged head pose.

Pipeline (all automatic):
  1. align each keyframe to the photo with SIFT features on the car/body around the head (scale+rotation+shift)
  2. cut out head+neck (GrabCut), match brightness/contrast to the photo's head
  3. add a mirrored copy of every keyframe (bald/symmetric heads mirror well), measure each head's yaw/pitch with landmarks
  4. drop near-duplicates, then per video frame pick the pose closest to the logged one (with hysteresis) and
     cross-dissolve between at most two heads; only rigid micro-motion, no warping
  5. only pixels inside the protected head/neck zone can change - everything else stays the original photo
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import landmarks
from .config import Geometry

CW, CH = 400, 420          # canonical crop size (k = 1)
COLLAR_Y, PIVOT = 335.0, (200.0, 335.0)
NANG = list(range(-120, 121, 10))
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def log(*a): print(*a, file=sys.stderr)


def _to_canon(img, k, interp=cv2.INTER_LANCZOS4):
  return img if abs(k - 1) < 1e-9 else cv2.resize(img, (CW, CH), interpolation=interp)


def _from_canon(img, size, interp=cv2.INTER_LINEAR):
  return img if img.shape[1] == size[0] and img.shape[0] == size[1] else cv2.resize(img, size, interpolation=interp)


def segment_head(c):
  """GrabCut head+neck mask on a canonical crop (head roughly centred, chin ~y 290)."""
  c8 = np.clip(c, 0, 255).astype(np.uint8); h, w = c8.shape[:2]
  m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
  m[:, :60] = cv2.GC_BGD; m[:, w - 50:] = cv2.GC_BGD; m[:40, :] = cv2.GC_BGD
  cv2.ellipse(m, (200, 200), (48, 70), 0, 0, 360, cv2.GC_PR_FGD, -1)
  cv2.ellipse(m, (200, 190), (30, 45), 0, 0, 360, cv2.GC_FGD, -1)
  cv2.rectangle(m, (180, 270), (220, 330), cv2.GC_FGD, -1)
  bg = np.zeros((1, 65)); fg = np.zeros((1, 65))
  cv2.grabCut(c8, m, None, bg, fg, 8, cv2.GC_INIT_WITH_MASK)
  out = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
  out[345:] = 0
  _, lab, _, _ = cv2.connectedComponentsWithStats(out)
  out = (lab == lab[190, 200]).astype(np.uint8) * 255
  out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
  ff = out.copy(); cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
  return out | ~ff


def outline_center(m):
  ys, xs = np.nonzero(m[:260]); cx, cy = xs.mean(), 165.0
  pts = []
  for a in NANG:
    r = math.radians(a); dx, dy = math.sin(r), -math.cos(r); t = 5; last = (cx, cy)
    while t < 260:
      x, y = cx + dx * t, cy + dy * t
      if x < 0 or y < 0 or x >= CW or y >= CH or m[int(y), int(x)] == 0:
        break
      last = (x, y); t += 1
    pts.append(last)
  return np.array(pts).mean(0)


def align(photo_gray, kf_bgr, g):
  """Similarity transform keyframe -> photo using SIFT on the scene around the head (head itself masked out)."""
  k, ex, ey = g.k, g.ex, g.ey
  sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.01)
  clahe = cv2.createCLAHE(3.0, (8, 8))
  mP = np.zeros(photo_gray.shape, np.uint8)
  x0, y0 = int(max(0, ex - 358 * k)), int(max(0, ey - 242 * k)); y1 = int(min(g.H, ey + 708 * k))
  mP[y0:y1, x0:] = 255
  cv2.ellipse(mP, (int(ex - 8 * k), int(ey + 8 * k)), (int(110 * k), int(150 * k)), 0, 0, 360, 0, -1)
  kP, dP = sift.detectAndCompute(clahe.apply(photo_gray), mP)
  kg = cv2.cvtColor(kf_bgr, cv2.COLOR_BGR2GRAY)
  kK, dK = sift.detectAndCompute(clahe.apply(kg), None)
  good = [a for a, b in cv2.BFMatcher().knnMatch(dK, dP, k=2) if a.distance < 0.8 * b.distance]
  if len(good) < 12:
    raise RuntimeError("too few matching features between keyframe and photo")
  src = np.float32([kK[a.queryIdx].pt for a in good]); dst = np.float32([kP[a.trainIdx].pt for a in good])
  M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6, maxIters=20000, confidence=0.999)
  if M is None or inl.sum() < 12:
    raise RuntimeError("could not align keyframe to photo")
  return M, int(inl.sum()), len(good)


class Keyframes:
  def __init__(self, photo_rgb, kf_dir, cfg, geom):
    self.g, self.cfg = geom, cfg["realhead"]
    g = geom
    X0, Y0, cw, ch = g.crop
    if X0 < 0 or Y0 < 0 or X0 + cw > g.W or Y0 + ch > g.H:
      raise ValueError("face anchor too close to the photo edge for realistic-head mode")
    photo_bgr = cv2.cvtColor(photo_rgb, cv2.COLOR_RGB2BGR)
    self.P = _to_canon(photo_bgr[Y0:Y0 + ch, X0:X0 + cw].astype(np.float32), g.k)
    pm = segment_head(self.P)
    heads = {"photo": (self.P, pm, "original photo")}
    files = sorted(p for p in Path(kf_dir).expanduser().iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
      raise FileNotFoundError(f"no keyframe images (.jpg/.png) in {kf_dir}")
    pg = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)
    self.align_info = {}
    for f in files:
      K = cv2.imread(str(f))
      try:
        M, ninl, ngood = align(pg, K, g)
      except RuntimeError as e:
        log(f"  ! skipping {f.name}: {e}"); continue
      s = math.hypot(M[0, 0], M[1, 0]); rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
      log(f"  aligned {f.name}: scale {s:.2f}, rotation {rot:+.1f} deg, {ninl}/{ngood} feature matches")
      Wp = cv2.warpAffine(K, M, (g.W, g.H), flags=cv2.INTER_LANCZOS4).astype(np.float32)
      c = _to_canon(Wp[Y0:Y0 + ch, X0:X0 + cw], g.k)
      m = segment_head(c)
      c2 = c.copy()                       # tone-match head to the photo's head
      for chn in range(3):
        a = c[..., chn][m > 0]; b = self.P[..., chn][pm > 0]
        c2[..., chn] = (c[..., chn] - a.mean()) / max(a.std(), 1e-3) * b.std() + b.mean()
      heads[f.stem] = (c2, m, f.name)
      heads[f.stem + "_mirror"] = (c2[:, ::-1].copy(), m[:, ::-1].copy(), f.name + " (mirrored)")
      self.align_info[f.stem] = dict(scale=s, rotation=rot, inliers=ninl)
    self.heads, self.meas = {}, {}
    for n, (img, m, label) in heads.items():
      lm = landmarks.detect(img)
      if lm is None:
        log(f"  ! no face found in {label}; skipped"); continue
      self.heads[n] = (img, m, label)
      self.meas[n] = dict(yaw=lm["yaw"], pitch=lm["pitch"], eyes=lm["pts"][[468, 473]], center=outline_center(m))
    if "photo" not in self.meas:
      raise RuntimeError("no face found in the photo's head area - check face.eye_mid / interocular_px")
    self._select()

  def _select(self):
    c, ph = self.cfg, self.meas["photo"]
    rel = {n: (v["yaw"], v["pitch"] - ph["pitch"]) for n, v in self.meas.items()}
    order = sorted((n for n in rel if n != "photo"), key=lambda n: -math.hypot(rel[n][0] - rel["photo"][0], rel[n][1]))
    keep = ["photo"]
    for n in order:
      y, p = rel[n]
      if abs(y - rel["photo"][0]) < c["near_photo_yaw_deg"] and abs(p) < c["dedupe_pitch_deg"]:
        continue
      if any(abs(y - rel[q][0]) < c["dedupe_yaw_deg"] and abs(p - rel[q][1]) < c["dedupe_pitch_deg"] for q in keep):
        continue
      keep.append(n)
    self.keys = {n: (rel[n][0], 1.0 if rel[n][1] >= c["down_keyframe_pitch_deg"] else 0.0) for n in keep}
    downs = [abs(y) for y, f in self.keys.values() if f > 0]
    self.down_span = max(downs) if downs else 0.0
    for n, v in self.meas.items():
      tag = ("used, head-down" if self.keys[n][1] else "used") if n in self.keys else "not used (duplicate pose)"
      log(f"  pose {self.heads[n][2]:34s} yaw {v['yaw']:+6.1f}  pitch {v['pitch'] - self.meas['photo']['pitch']:+5.1f} vs photo  -> {tag}")

  def sheet(self, path, photo_rgb):
    """Contact sheet of every head composited onto the photo, with measured angles."""
    font = ImageFont.truetype(str(Path(__file__).parent / "fonts" / "Inter-SemiBold.ttf"), 17)
    names = list(self.heads); tw, th = 300, 345
    cols = 4; rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 10) + 10, rows * (th + 64) + 10), (12, 13, 16)); d = ImageDraw.Draw(sheet)
    comp = HeadCompositor(self, None, 0, 0, photo_rgb, dry=True)
    for i, n in enumerate(names):
      img = comp.static(n)
      X0, Y0, cw, ch = self.g.crop
      tile = Image.fromarray(img[Y0:Y0 + ch, X0:X0 + cw]).resize((tw, th))
      x = 10 + (i % cols) * (tw + 10); y = 10 + (i // cols) * (th + 64)
      sheet.paste(tile, (x, y))
      v = self.meas[n]; used = n in self.keys
      d.text((x, y + th + 4), self.heads[n][2][:34], font=font, fill=(255, 255, 255))
      d.text((x, y + th + 26), f"yaw {v['yaw']:+.1f}  pitch {v['pitch'] - self.meas['photo']['pitch']:+.1f}  " + ("USED" if used else "unused"),
             font=font, fill=(120, 220, 150) if used else (200, 160, 90))
    sheet.save(path)


class HeadCompositor:
  def __init__(self, kfs, route, start, duration, photo_rgb, fps=30, vignette=None, dry=False):
    self.kf, self.g, self.c = kfs, kfs.g, kfs.cfg
    g = self.g
    X0, Y0, cw, ch = g.crop
    self.RAW = np.asarray(photo_rgb).astype(np.float64)
    self.vig = vignette if vignette is not None else np.ones((g.H, g.W, 1))
    poly = (np.array(g.allow_poly) - [X0, Y0]) / g.k
    al = np.zeros((CH, CW), np.uint8); cv2.fillPoly(al, [np.round(poly).astype(np.int32)], 255)
    self.allow_soft = cv2.GaussianBlur(cv2.erode(al, np.ones((9, 9), np.uint8)).astype(np.float32) / 255, (0, 0), 2.0) * (al > 0)
    alp = np.zeros((ch, cw), np.uint8); cv2.fillPoly(alp, [np.round(np.array(g.allow_poly) - [X0, Y0]).astype(np.int32)], 255)
    self.allow_hard = alp > 0                       # in photo pixels: nothing outside ever changes
    yy = np.arange(CH)[:, None].astype(np.float32)
    col = np.clip((COLLAR_Y - yy) / 40.0, 0, 1); self.collar = col * col * (3 - 2 * col)
    pm = kfs.heads["photo"][1]
    crop8 = kfs.P[..., ::-1].astype(np.uint8)
    ipm = cv2.dilate(pm, np.ones((9, 9), np.uint8)); ipm[al == 0] = 0
    self.clean = cv2.inpaint(crop8, ipm, 25, cv2.INPAINT_NS).astype(np.float64)
    self.oldhead = cv2.GaussianBlur(cv2.dilate(pm, np.ones((5, 5), np.uint8)).astype(np.float32) / 255, (0, 0), 2.0) * self.allow_soft * self.collar
    self.grain = cv2.GaussianBlur(np.random.default_rng(7).normal(0, 4.0, (CH, CW)).astype(np.float32), (0, 0), 0.6)[..., None]
    self.raw_crop = _to_canon(self.RAW[Y0:Y0 + ch, X0:X0 + cw].astype(np.float32), g.k).astype(np.float64)
    if dry:
      return
    self.r, self.start, self.fps = route, float(start), fps
    self.nf = int(round(duration * fps))
    self.base_y, self.base_p, self.base_r = route.baseline(self.start, self.start + duration)
    self.sched = self._schedule()

  # ---------------------------------------------------------------- pose -> keyframe schedule
  def pose(self, t):
    r = self.r
    sm = lambda a: float(np.interp(t, r.tds, self._smooth(a)))
    yn, pn, rn = sm(r.yaw_net), sm(r.pitch_net), sm(r.roll_net)
    Y = self.kf.keys["photo"][0] + (yn - self.base_y)
    c = self.c
    f = float(np.clip((-(pn - self.base_p) - c["head_down_start_deg"]) / (c["head_down_full_deg"] - c["head_down_start_deg"]), 0, 1))
    f *= float(np.clip(1 - (abs(Y) - self.kf.down_span) / 10.0, 0, 1))
    return Y, f, rn - self.base_r

  _sm_cache = {}
  def _smooth(self, a, sig=2):
    key = id(a)
    if key not in self._sm_cache:
      kk = np.exp(-0.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2); kk /= kk.sum()
      self._sm_cache[key] = np.convolve(np.pad(a, (3 * sig, 3 * sig), mode="edge"), kk, "valid")
    return self._sm_cache[key]

  def _schedule(self):
    keys, c = self.kf.keys, self.c
    cost = lambda n, Y, f: abs(Y - keys[n][0]) + 30 * abs(f - keys[n][1])
    XF = max(1, int(round(c["dissolve_s"] * self.fps))); dwell = int(round(c["min_hold_s"] * self.fps))
    Y, f, _ = self.pose(self.start)
    cur = min(keys, key=lambda n: cost(n, Y, f)); prev = None; t_sw = -10 ** 6; out = []
    for i in range(self.nf):
      Y, f, _ = self.pose(self.start + i / self.fps)
      best = min(keys, key=lambda n: cost(n, Y, f))
      if best != cur and i - t_sw >= dwell and cost(cur, Y, f) - cost(best, Y, f) > c["hysteresis_deg"]:
        prev, cur, t_sw = cur, best, i
      s = min(1.0, (i - t_sw) / XF) if prev is not None else 1.0
      s = s * s * (3 - 2 * s)
      out.append({cur: 1.0} if s >= 1 else {prev: 1 - s, cur: s})
    return out

  # ---------------------------------------------------------------- compositing
  def _layer(self, n, Y=None, r=0.0):
    img, m, _ = self.kf.heads[n]
    c = self.c
    dx = 0.0 if Y is None else float(np.clip(0.25 * (Y - self.kf.keys[n][0]), -c["micro_shift_px"], c["micro_shift_px"]))
    th = float(np.clip(r, -c["micro_roll_deg"], c["micro_roll_deg"]))
    M = cv2.getRotationMatrix2D(PIVOT, -th, 1.0); M[0, 2] += dx
    H_ = cv2.warpAffine(img.astype(np.float32), M, (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., ::-1].astype(np.float64)
    m_ = cv2.warpAffine((m > 0).astype(np.float32), M, (CW, CH), flags=cv2.INTER_LINEAR)
    a = cv2.GaussianBlur(m_, (0, 0), 1.6) * self.allow_soft * self.collar
    base = self.raw_crop * (1 - self.oldhead[..., None]) + self.clean * self.oldhead[..., None]
    comp = base * (1 - a[..., None]) + (H_ + self.grain) * a[..., None]
    meas = self.kf.meas[n]
    e = meas["eyes"] @ M[:, :2].T + M[:, 2]; o = meas["center"] @ M[:, :2].T + M[:, 2]
    return comp, e, o

  def _paste(self, crop_canon):
    g = self.g; X0, Y0, cw, ch = g.crop
    out = self.RAW.copy()
    crop = _from_canon(crop_canon, (cw, ch))
    out[Y0:Y0 + ch, X0:X0 + cw] = np.where(self.allow_hard[..., None], crop, self.RAW[Y0:Y0 + ch, X0:X0 + cw])
    return out

  def static(self, n):
    if n == "photo":
      return self.RAW.astype(np.uint8)
    comp, _, _ = self._layer(n)
    return np.clip(self._paste(comp), 0, 255).astype(np.uint8)

  def frame(self, t, fi):
    g = self.g; X0, Y0, cw, ch = g.crop
    w = self.sched[min(fi, self.nf - 1)]
    Y, f, r = self.pose(t)
    ph = self.kf.meas["photo"]
    if list(w) == ["photo"]:
      out = self.RAW
      E, O = ph["eyes"].copy(), ph["center"].copy()
    else:
      crop = np.zeros((CH, CW, 3)); E = np.zeros((2, 2)); O = np.zeros(2)
      for n, wk in w.items():
        if n == "photo":
          c_, e, o = self.raw_crop, ph["eyes"], ph["center"]
        else:
          c_, e, o = self._layer(n, Y, r)
        crop += wk * c_; E += wk * e; O += wk * o
      out = self._paste(crop)
    img = Image.fromarray(np.clip(out * self.vig, 0, 255).astype(np.uint8))
    E = E * g.k + [X0, Y0]; E = E[np.argsort(E[:, 0])]
    sh = (O - ph["center"]) * g.k
    return img, dict(eyes=E, shift=(float(sh[0]), float(sh[1])), weights=w)
