"""Steering wheel that turns with the logged steering angle (v2).

The photo's wheel is treated as a flat disc seen in perspective. Its outline is an ellipse (calibrated like the face
anchor); an affine map takes the unit circle in the wheel's own plane ("frontal" view, as the camera sees it) onto
that ellipse, so a rotation in the wheel's plane becomes `H . Rot(q) . H^-1` in the photo.

By default (since the v2 review) the photo is NOT changed: the wheel's angle is shown as an indicator ring on the
rim (hands masked out) plus the panel. `--warp-spokes` (experimental) turns on the pixel spoke rotation below; it
can cut through the hands and smear the rim on photos like the mannequin, so it is off by default.

What moves and what doesn't (--warp-spokes):
  * hub + spokes rotate: they are cut out of the photo in the frontal view (hub disc + spoke wedges, minus see-through
    gaps that are clearly brighter), a hidden spoke can be filled with a mirrored copy of a visible one, and the
    pixels they uncover are inpainted once (clean plate)
  * the rim is rotationally symmetric, so it stays the original photo (it looks identical when turned)
  * the hands stay where they are, in front of the wheel (pixel-for-pixel the photo); the rim slides under them
  * everything else, including the realistic-head zone, is never touched
At 0 deg the output is exactly the photo.

Sign: openpilot's steeringAngleDeg is + = left (counter-clockwise from the driver's seat). The driver camera faces the
driver, i.e. it sees the wheel from behind, so a left turn is CLOCKWISE in the photo
(`wheel.positive_angle_is_clockwise_in_image`, default true).
"""
import math

import cv2
import numpy as np

from .config import Geometry


# ------------------------------------------------------------------ geometry
def ellipse_frame(ellipse, up_dir=(0.0, -1.0)):
  """Affine frame of a wheel ellipse. Returns (c, A): image point = c + A @ u for u on the unit circle in the
  wheel plane (u = (x right, y down) as seen by the camera; (0,-1) = 12 o'clock maps to `up_dir` from the centre).
  `ellipse` uses OpenCV's fitEllipse convention: (cx, cy, full_width, full_height, angle_deg). det(A) > 0, so the
  camera's handedness is kept (clockwise in the frontal view is clockwise in the photo)."""
  cx, cy, w, h, ang = map(float, ellipse)
  t = math.radians(ang)
  A0 = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]) @ np.diag([w / 2, h / 2])
  ss = np.linspace(0, 2 * math.pi, 7200, endpoint=False)
  P = A0 @ np.stack([np.cos(ss), np.sin(ss)])
  ud = np.asarray(up_dir, float); ud = ud / np.linalg.norm(ud)
  s_up = ss[int(np.argmax((P.T @ ud) / np.linalg.norm(P, axis=0)))]
  up = A0 @ [math.cos(s_up), math.sin(s_up)]
  for sg in (1, -1):
    right = A0 @ [math.cos(s_up + sg * math.pi / 2), math.sin(s_up + sg * math.pi / 2)]
    A = np.c_[right, -up]
    if np.linalg.det(A) > 0:
      return np.array([cx, cy]), A
  raise ValueError("degenerate wheel ellipse")


def rot(q_deg):
  """Clockwise (on screen, y down) rotation by q degrees."""
  q = math.radians(q_deg)
  return np.array([[math.cos(q), -math.sin(q)], [math.sin(q), math.cos(q)]])


def screen_angle(angle_deg, clockwise=True):
  """Logged steering angle (+ = left) -> clockwise rotation of the wheel in the photo."""
  return float(angle_deg) if clockwise else -float(angle_deg)


def rim_point(c, A, q_deg, r=1.0):
  """Photo position of the wheel-plane point at clock angle q (deg clockwise from 12 o'clock, camera view), radius r."""
  q = math.radians(q_deg)
  return c + A @ (r * np.array([math.sin(q), -math.cos(q)]))


# ------------------------------------------------------------------ auto calibration
def _nelder_mead(f, x0, step, iters=400, tol=1e-4):
  n = len(x0); pts = [np.asarray(x0, float)]
  for i in range(n):
    p = np.array(x0, float); p[i] += step[i]; pts.append(p)
  vals = [f(p) for p in pts]
  for _ in range(iters):
    o = np.argsort(vals); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
    if abs(vals[-1] - vals[0]) < tol:
      break
    c = np.mean(pts[:-1], 0); xr = c + (c - pts[-1]); fr = f(xr)
    if fr < vals[0]:
      xe = c + 2 * (c - pts[-1]); fe = f(xe)
      pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
    elif fr < vals[-2]:
      pts[-1], vals[-1] = xr, fr
    else:
      xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
      if fc < vals[-1]:
        pts[-1], vals[-1] = xc, fc
      else:
        pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]; vals = [f(p) for p in pts]
  i = int(np.argmin(vals)); return pts[i], vals[i]


def find_wheel(img_bgr, cfg, seed=0):
  """Find the wheel rim ellipse near the position predicted from the face anchor (the mannequin photo's wheel
  relative to its face, scaled by interocular distance), refined to the dark rim's outer edge.
  Returns ([cx, cy, w, h, angle_deg], score) - check the calibration preview."""
  H, W = img_bgr.shape[:2]
  f = cfg["face"]; k = float(f["interocular_px"]) / Geometry.REF_IOD; ex, ey = map(float, f["eye_mid"])
  dx, dy, w0, h0, a0 = cfg["wheel"]["ellipse_rel"]
  if cfg["wheel"].get("_mirror_x"):
    dx, a0 = -dx, -a0
  prior = np.array([ex + dx * k, ey + dy * k, w0 * k / 2, h0 * k / 2, math.radians(a0)])
  g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
  gb = cv2.GaussianBlur(255 * (g / max(float(np.percentile(g, 99)), 1.0)) ** 0.5, (0, 0), 2.0)
  S = np.linspace(0, 2 * math.pi, 360, endpoint=False); cs, sn = np.cos(S), np.sin(S)

  def samp(p, kk):
    cx, cy, a, b, t = p
    x = cx + kk * (a * cs * math.cos(t) - b * sn * math.sin(t)); y = cy + kk * (a * cs * math.sin(t) + b * sn * math.cos(t))
    ok = (x >= 1) & (x < W - 2) & (y >= 1) & (y < H - 2)
    v = cv2.remap(gb, x.astype(np.float32)[None], y.astype(np.float32)[None], cv2.INTER_LINEAR).ravel()
    return v, ok

  lim = np.array([0.6 * 56 * k, 0.6 * 56 * k, 0.15 * prior[2], 0.15 * prior[3], math.radians(10)])
  def score(p):        # dark rim just inside the outline, brighter just outside; stay near the prior
    if np.any(np.abs(p - prior) > lim):
      return 0.0
    vo, o1 = samp(p, 1.04); vr, o2 = samp(p, 0.97)
    ok = o1 & o2
    return -float(np.sum(np.where(ok, np.clip(vo - vr, 0, 60), 0)) / len(S))
  rng = np.random.default_rng(seed)
  best = (score(prior), prior)
  for i in range(8):
    p0 = prior if i == 0 else prior + rng.uniform(-0.5, 0.5, 5) * lim
    x, fv = _nelder_mead(score, p0, [0.2 * lim[0], 0.2 * lim[1], 0.2 * lim[2], 0.2 * lim[3], 0.2 * lim[4]])
    if fv < best[0]:
      best = (fv, x)
  cx, cy, a, b, t = best[1]
  return [round(float(cx), 1), round(float(cy), 1), round(float(2 * a), 1), round(float(2 * b), 1), round(math.degrees(t), 1)], -best[0]


# ------------------------------------------------------------------ compositor
class WheelCompositor:
  """Per-frame: rotate the photo's hub+spokes by the logged steering angle (see module doc)."""

  def __init__(self, base_rgb, cfg, ss=1.0, warp_spokes=False):
    wc = cfg["wheel"]
    if not wc.get("ellipse"):
      raise ValueError("wheel.ellipse is not set - run `drivermover calibrate --image <photo> --auto` (or use --no-wheel)")
    self.cfg = wc
    self.cw = bool(wc.get("positive_angle_is_clockwise_in_image", True))
    self.base = np.asarray(base_rgb).astype(np.float32)
    H, W = self.base.shape[:2]
    self.c, self.A = ellipse_frame(wc["ellipse"], wc.get("up_dir", (0.0, -1.0)))
    self.Rf = Rf = int(math.ceil(max(np.linalg.norm(self.A, axis=0)) * ss))   # frontal pixels per unit radius
    self.S = S = 2 * Rf + 8; self.o = o = S / 2.0
    # frontal canvas <- photo
    self.M_f2i = np.c_[self.A / Rf, self.c - self.A @ [o, o] / Rf]
    warp_in = lambda im, interp=cv2.INTER_CUBIC: cv2.warpAffine(im, self.M_f2i, (S, S), flags=interp | cv2.WARP_INVERSE_MAP, borderValue=0)
    F = warp_in(self.base)
    valid = warp_in(np.ones((H, W), np.float32), cv2.INTER_LINEAR) > 0.999
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    r = np.hypot(xx - o, yy - o) / Rf
    q = np.degrees(np.arctan2(xx - o, -(yy - o))) % 360.0
    self.r_f, self.q_f = r, q
    # hands (bright, large blobs on/around the rim) stay in front
    g = cv2.cvtColor(self.base.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hz = np.zeros((H, W), np.uint8)
    cx, cy, w, h, ang = wc["ellipse"]
    cv2.ellipse(hz, ((cx, cy), (w * 1.25, h * 1.25), ang), 255, -1)
    hm = ((g > wc["hand_threshold"]) & (hz > 0)).astype(np.uint8)
    hm = cv2.morphologyEx(hm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(hm)
    k2 = (w * h) / (331.0 * 508.0)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= wc["hand_min_area"] * k2
    hm = keep[lab].astype(np.uint8)
    hm = cv2.dilate(hm, np.ones((5, 5), np.uint8))
    hb = cv2.GaussianBlur(hm.astype(np.float32), (0, 0), 1.2)
    self.hand = np.where(hb > 0.98, 1.0, hb).astype(np.float32)[..., None]   # hand cores stay exactly the photo
    # overlay mask: the rim indicator is never drawn over the hands/arms (a bit wider than the hand cores)
    self.hand_overlay = cv2.GaussianBlur(cv2.dilate(hm, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 1.5)
    self.warp_spokes = bool(warp_spokes)
    if not self.warp_spokes:        # default: the photo is not changed at all, only the overlay indicator is drawn
      return
    hand_f = warp_in(hm.astype(np.float32), cv2.INTER_LINEAR) > 0.2
    # moving parts in the frontal view
    gf = cv2.GaussianBlur(cv2.cvtColor(np.clip(F, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2.0)
    rim_in = float(wc["rim_inner"]); hub_r = float(wc["hub_radius"])
    spokes = np.zeros((S, S), bool)
    for qa, hw in wc["spokes"]:
      d = np.abs((q - qa + 180) % 360 - 180)
      spokes |= (d <= hw) & (r >= hub_r * 0.9) & (r <= rim_in + 0.02)
    gap = gf > float(np.percentile(gf[(r < rim_in) & valid & ~hand_f], 100 * wc["gap_quantile"]))
    gap = cv2.morphologyEx(gap.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    mov = spokes & (r > hub_r) & ~gap & valid
    mov = cv2.morphologyEx(mov.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    # layer pixels: inpaint the hands out of the moving parts
    F8 = np.clip(F, 0, 255).astype(np.uint8)
    hole = (cv2.dilate(hand_f.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & mov
    if hole.any():
      F8 = cv2.inpaint(F8, hole.astype(np.uint8), 9, cv2.INPAINT_TELEA)
    layer = F8.astype(np.float32); alpha = mov.astype(np.float32)
    # hidden spokes: mirrored copies of visible ones
    for qs, qd in wc.get("mirror_spokes", []):
      m = math.radians((qs + qd) / 2.0)
      u = np.array([math.sin(m), -math.cos(m)])          # mirror axis direction (frontal)
      Rm = 2 * np.outer(u, u) - np.eye(2)
      Mm = np.c_[Rm, np.array([o, o]) - Rm @ [o, o]]
      src_w = next(hw for qa, hw in wc["spokes"] if abs(((qa - qs + 180) % 360) - 180) < 1e-6)
      d = np.abs((q - qd + 180) % 360 - 180)
      dst = (d <= src_w) & (r > hub_r) & (r <= rim_in + 0.02)
      mL = cv2.warpAffine(layer, Mm, (S, S), flags=cv2.INTER_LINEAR)
      mA = cv2.warpAffine(alpha, Mm, (S, S), flags=cv2.INTER_LINEAR) * dst
      layer = np.where(mA[..., None] > alpha[..., None], mL, layer); alpha = np.maximum(alpha, mA)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0) * (r <= rim_in + 0.03)
    self.layer, self.alpha = layer, alpha
    self.mov_f, self.gap_f, self.hand_f = mov, gap, hand_f
    # ROI in the photo (where anything can change) and the clean plate behind the moving parts
    corners = np.array([self.c + self.A @ v * (rim_in + 0.05) for v in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
    x0, y0 = np.floor(corners.min(0)).astype(int); x1, y1 = np.ceil(corners.max(0)).astype(int)
    self.roi = (max(0, x0), max(0, y0), min(W, x1), min(H, y1))
    X0, Y0, X1, Y1 = self.roi
    a0 = self._warp(self.alpha, 0.0)
    m0 = (cv2.dilate((a0 > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)))
    cleanF = self._angular_fill(F, cv2.dilate((self.alpha > 0.02).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0, valid & ~hand_f)
    self.clean = self._warp(cleanF, 0.0, cv2.INTER_CUBIC)
    roi = self.base[Y0:Y1, X0:X1]
    self.clean = np.where(m0[..., None] > 0, self.clean, roi).astype(np.float32)
    self.m0 = m0.astype(bool)
    # hard limit: nothing outside the rim's inner edge can ever change (rim, hands on the rim, body, head, HUD area)
    lim = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    pts = np.array([rim_point(self.c, self.A, a, rim_in + 0.03) - [X0, Y0] for a in np.arange(0, 360, 1.0)])
    cv2.fillPoly(lim, [np.round(pts).astype(np.int32)], 1)
    self.allow = lim.astype(bool)[..., None]
    self.hand_roi = self.hand[Y0:Y1, X0:X1]

  def _angular_fill(self, F, hole, valid):
    """Fill `hole` in the frontal view by interpolating along circles (same radius) from the visible background
    on either side - what is behind a spoke is guessed from what is next to it at the same distance from the hub."""
    S, o = self.S, self.o
    nr, na = int(self.Rf * 1.0), 1440
    flags = cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR
    P = cv2.warpPolar(F, (nr, na), (o, o), self.Rf, flags)
    Hm = cv2.warpPolar(hole.astype(np.float32), (nr, na), (o, o), self.Rf, flags) > 0.01
    Vm = cv2.warpPolar(valid.astype(np.float32), (nr, na), (o, o), self.Rf, flags) > 0.99
    out = P.copy(); idx = np.arange(na)
    for j in range(nr):
      good = ~Hm[:, j] & Vm[:, j]
      if good.sum() < 8 or Hm[:, j].sum() == 0:
        continue
      gi = idx[good]
      for ch in range(P.shape[2]):
        out[Hm[:, j], j, ch] = np.interp(idx[Hm[:, j]], gi, P[gi, j, ch], period=na)
    sm = cv2.GaussianBlur(out, (0, 0), 1.5)
    out = np.where(Hm[..., None], sm, out)
    back = cv2.warpPolar(out, (S, S), (o, o), self.Rf, flags | cv2.WARP_INVERSE_MAP)
    noise = cv2.GaussianBlur(np.random.default_rng(3).normal(0, 1.5, (S, S)).astype(np.float32), (0, 0), 0.8)[..., None]
    back = back * float(self.cfg.get("behind_shade", 0.8)) + noise    # behind the wheel is in its shadow
    return np.where(hole[..., None], back, F).astype(np.float32)

  def _M(self, q_deg):
    X0, Y0 = self.roi[:2]
    A = self.A @ rot(q_deg) / self.Rf
    return np.c_[A, self.c - A @ [self.o, self.o] - [X0, Y0]]

  def _warp(self, im, q_deg, interp=cv2.INTER_LINEAR):
    X0, Y0, X1, Y1 = self.roi
    return cv2.warpAffine(im, self._M(q_deg), (X1 - X0, Y1 - Y0), flags=interp, borderValue=0)

  def apply(self, frame_rgb, angle_deg):
    """frame_rgb: HxWx3 uint8/float (photo, possibly with the realistic head). Returns uint8 array."""
    q = screen_angle(angle_deg, self.cw)
    out = np.array(frame_rgb, dtype=np.uint8, copy=True)
    if not self.warp_spokes:
      return out
    X0, Y0, X1, Y1 = self.roi
    a = self._warp(self.alpha, q)[..., None]
    L = self._warp(self.layer, q, cv2.INTER_CUBIC)
    cur = out[Y0:Y1, X0:X1].astype(np.float32)
    comp = np.where(self.m0[..., None], self.clean, cur)
    comp = comp * (1 - a) + L * a
    comp = comp * (1 - self.hand_roi) + cur * self.hand_roi
    comp = np.where(self.allow, comp, cur)
    out[Y0:Y1, X0:X1] = np.clip(comp, 0, 255).astype(np.uint8)
    return out

  # ---------------------------------------------------------------- debugging / calibration preview
  def frontal_debug(self):
    """Frontal (wheel-plane) view, brightened, with moving parts tinted and gaps/hands outlined."""
    F = cv2.warpAffine(self.base, self.M_f2i, (self.S, self.S), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    v = np.clip(255 * (np.clip(F, 0, None) / 60.0) ** 0.8, 0, 255)
    v[self.mov_f] = v[self.mov_f] * 0.55 + np.array([0, 90, 160]) * 0.45
    v[self.alpha > 0.5] = v[self.alpha > 0.5] * 0.8 + np.array([0, 60, 120]) * 0.2
    v = v.astype(np.uint8).copy()
    cs_, _ = cv2.findContours(self.gap_f.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(v, cs_, -1, (255, 220, 0), 1)
    cs_, _ = cv2.findContours(self.hand_f.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(v, cs_, -1, (255, 60, 200), 1)
    o, Rf = int(self.o), self.Rf
    for rr in (self.cfg["hub_radius"], self.cfg["rim_inner"], 1.0):
      cv2.circle(v, (o, o), int(rr * Rf), (0, 200, 0), 1)
    for k_ in range(0, 360, 30):
      qq = math.radians(k_)
      cv2.line(v, (o, o), (int(o + Rf * math.sin(qq)), int(o - Rf * math.cos(qq))), (160, 80, 0), 1)
    return v
