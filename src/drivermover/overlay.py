"""Draw the driver-monitoring overlay (face wireframe, gaze, HUD panels) and encode the video."""
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import Geometry
from .data import AWARE_PRE, AWARE_PROMPT, PITCH_TH, YAW_TH

FONT_DIR = Path(__file__).parent / "fonts"
SS = 2          # supersampling for smooth lines
FPS = 30
REF_W, REF_H = 1152, 1728   # HUD layout reference (2:3 portrait)
GREEN = (51, 209, 122); AMBER = (255, 176, 32); ORANGE = (218, 111, 37); RED = (230, 60, 60); WHITE = (255, 255, 255)
CYAN = (80, 200, 255); VIOLET = (190, 140, 255)


class T:
  """Maps layout coordinates to supersampled canvas pixels."""
  def __init__(self, scale=1.0, ox=0.0, oy=0.0):
    self.u, self.ox, self.oy = scale, ox, oy
  def x(self, v): return (self.ox + v * self.u) * SS
  def y(self, v): return (self.oy + v * self.u) * SS
  def s(self, v): return v * self.u * SS
  def p(self, xy): return (self.x(xy[0]), self.y(xy[1]))


def _font(weight, size):
  return ImageFont.truetype(str(FONT_DIR / f"Inter-{weight}.ttf"), max(6, int(size)))


class Fonts:
  SPEC = dict(h1=("Bold", 46), h2=("SemiBold", 30), b=("Medium", 22), sm=("Medium", 18), xs=("Regular", 15),
              big=("Bold", 64), num=("SemiBold", 34), lab=("SemiBold", 15))
  def __init__(self, scale):
    self.f = {k: _font(w, s * SS * scale) for k, (w, s) in self.SPEC.items()}
  def __getitem__(self, k): return self.f[k]


def panel(dr, tr, box, r=26, fill=(14, 16, 20, 165), outline=(255, 255, 255, 28)):
  dr.rounded_rectangle([tr.x(box[0]), tr.y(box[1]), tr.x(box[2]), tr.y(box[3])], radius=int(tr.s(r)), fill=fill,
                       outline=outline, width=max(1, int(tr.s(1.5))))
def text(dr, tr, xy, t, font, fill=WHITE, a=255, anchor="la"):
  dr.text(tr.p(xy), t, font=font, fill=fill + (int(a),), anchor=anchor)
def line(dr, tr, pts, c, a, w):
  dr.line([tr.p(q) for q in pts], fill=c + (int(a),), width=max(1, int(tr.s(w))), joint="curve")
def rrect(dr, tr, box, r, **kw):
  dr.rounded_rectangle([tr.x(box[0]), tr.y(box[1]), tr.x(box[2]), tr.y(box[3])], radius=int(tr.s(r)), **kw)
def ellipse(dr, tr, cx, cy, rx, ry, **kw):
  dr.ellipse([tr.x(cx) - tr.s(rx), tr.y(cy) - tr.s(ry), tr.x(cx) + tr.s(rx), tr.y(cy) + tr.s(ry)], **kw)
def mix(c1, c2, f): return tuple(int(c1[i] + (c2[i] - c1[i]) * f) for i in range(3))


# ------------------------------------------------------------ 3D head model (mm). camera frame: x right, y down, z into scene
RX, RY, RZ, CY = 72., 110., 92., 5.
FOC = 650.0
def surf(x, y):
  return -RZ * math.sqrt(max(1 - (x / RX) ** 2 - ((y - CY) / RY) ** 2, 0))
def rotm(yaw, pitch, roll):
  y, p, r = map(math.radians, (-yaw, -pitch, roll))
  Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
  Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
  Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
  return Rz @ Rx @ Ry
EYES = np.array([[-32, 0, surf(-32, 0) + 10], [32, 0, surf(32, 0) + 10]])
def _mesh():
  L = []
  sph = lambda lon, lat: (RX * math.cos(lat) * math.sin(lon), CY + RY * math.sin(lat), -RZ * math.cos(lat) * math.cos(lon))
  for lon in range(-75, 76, 15):
    L.append(np.array([sph(math.radians(lon), math.radians(b)) for b in np.linspace(-80, 80, 33)]))
  for lat in range(-60, 76, 15):
    L.append(np.array([sph(math.radians(a), math.radians(lat)) for a in np.linspace(-90, 90, 37)]))
  return L
MESH = _mesh()
def _feat(pts): return np.array([(x, y, surf(x, y) + o) for x, y, o in pts])
FEATS = [_feat([(-46, -16, 4), (-32, -21, 0), (-18, -17, 0)]), _feat([(18, -17, 0), (32, -21, 0), (46, -16, 4)]),
         _feat([(0, -12, 0), (0, 10, -6), (-2, 36, -18), (0, 40, -20), (9, 38, -10)]),
         _feat([(-20, 70, 0), (0, 73, -2), (20, 70, 0)]),
         _feat([(-60, 20, 0), (-55, 60, 0), (-38, 95, 0), (0, 112, 0), (38, 95, 0), (55, 60, 0), (60, 20, 0)])]


class Renderer:
  def __init__(self, route, image_path, cfg, start, duration=15.0, head=None, subtle=False):
    self.r, self.cfg, self.start, self.end = route, cfg, float(start), float(start) + float(duration)
    self.nf = int(round(duration * FPS))
    self.photo = Image.open(image_path).convert("RGB")
    self.W, self.H = self.photo.size
    self.g = Geometry(cfg, (self.W, self.H))
    self.mph = cfg["hud"]["speed_units"].lower() != "kmh"
    yy, xx = np.mgrid[0:self.H, 0:self.W]
    rr = np.sqrt(((xx - self.W / 2) / (self.W / 2)) ** 2 + ((yy - self.H / 2) / (self.H / 2)) ** 2)
    self.vig = np.clip(1 - 0.22 * np.clip(rr - 0.75, 0, None) ** 1.5, 0.75, 1)[..., None] if cfg["hud"]["vignette"] else np.ones((self.H, self.W, 1))
    self.base = Image.fromarray((np.asarray(self.photo).astype(float) * self.vig).astype(np.uint8))
    self.base_y, self.base_p, self.base_r = route.baseline(self.start, self.end) if route is not None else (0.0, 0.0, 0.0)
    dm = cfg["data_mapping"]
    self.ysgn = 1.0 if dm["positive_yaw_is_image_right"] else -1.0
    self.psgn = 1.0 if dm["positive_pitch_is_up"] else -1.0
    # wireframe anchor: with the photo's apparent pose the model's eye midpoint lands on eye_mid
    self.scale = 0.894 * self.g.k
    m = self.proj((rotm(self.g.apparent_yaw, self.g.apparent_pitch, 0) @ EYES.T).T, (0, 0)).mean(0)
    self.center = (self.g.ex - m[0], self.g.ey - m[1])
    u = min(self.W / REF_W, self.H / REF_H)
    self.hud = T(u, (self.W - REF_W * u) / 2, (self.H - REF_H * u) / 2)
    self.face = T(1.0)
    self.fonts = Fonts(u)
    self.ffonts = Fonts(self.g.k)
    self.head = head            # realistic-head compositor (or None)
    self.subtle = subtle or head is not None

  def proj(self, P, C):
    k = FOC / (FOC + P[..., 2])
    return np.stack([C[0] + self.scale * P[..., 0] * k, C[1] + self.scale * P[..., 1] * k], -1)

  def state_color(self, st):
    if st["a2"]: return ORANGE
    if st["a1"] or st["dist"]: return AMBER
    return GREEN

  # ------------------------------------------------------------------ face layer
  def draw_face(self, dr, st, hp):
    g, tr, k = self.g, self.face, self.g.k
    yaw = g.apparent_yaw + self.ysgn * (st["yn"] - self.base_y)
    pitch = g.apparent_pitch + self.psgn * (st["pn"] - self.base_p)
    R = rotm(yaw, pitch, st["rn"])
    C = (self.center[0] + hp.get("shift", (0, 0))[0], self.center[1] + hp.get("shift", (0, 0))[1])
    sc = self.state_color(st)
    bx = g.face_box; bc = GREEN if st["fdet"] else RED; L = 28 * k
    for (x, y, dx, dy) in [(bx[0], bx[1], 1, 1), (bx[2], bx[1], -1, 1), (bx[0], bx[3], 1, -1), (bx[2], bx[3], -1, -1)]:
      line(dr, tr, [(x + dx * L, y), (x, y), (x, y + dy * L)], bc, 230, 3 * k)
    panel(dr, tr, (bx[0], bx[1] - 42 * k, bx[2], bx[1] - 8 * k), r=10 * k, fill=(14, 16, 20, 170))
    cy = bx[1] - 25 * k
    text(dr, tr, (bx[0] + 12 * k, cy), "FACE", self.ffonts["lab"], bc, 255, "lm")
    text(dr, tr, (bx[2] - 12 * k, cy), f"{st['face'] * 100:.0f}%", self.ffonts["lab"], WHITE, 255, "rm")
    ma, mw, fa, fw = ((14, 32, 1.0), None, 70, 1.3) if self.subtle else ((40, 70, 1.2), None, 170, 2)
    for pts in MESH:
      P = (R @ pts.T).T
      n = (R @ np.c_[pts[:, 0] / RX ** 2, (pts[:, 1] - CY) / RY ** 2, pts[:, 2] / RZ ** 2].T).T
      vis = -n[:, 2] / np.linalg.norm(n, axis=1)
      p2 = self.proj(P, C)
      for i in range(len(p2) - 1):
        v = min(vis[i], vis[i + 1])
        if v > 0.05:
          line(dr, tr, [p2[i], p2[i + 1]], sc, ma[0] + ma[1] * v, ma[2] * k)
    for f in FEATS:
      line(dr, tr, list(map(tuple, self.proj((R @ f.T).T, C))), WHITE, fa, fw * k)
    fwd = R @ np.array([0, 0, -1.])
    E = hp["eyes"] if "eyes" in hp else self.proj((R @ EYES.T).T, C)
    oa, ow, da = (175, 1.6, 190) if self.subtle else (240, 2, 240)
    for (ex, ey), eo, bl in [(E[0], st["re"], st["rb"]), (E[1], st["le"], st["lb"])]:
      end = (ex + fwd[0] * self.scale * 520, ey + fwd[1] * self.scale * 520)
      for s_ in range(14):
        a0, a1 = s_ / 14, (s_ + 1) / 14
        line(dr, tr, [(ex + (end[0] - ex) * a0, ey + (end[1] - ey) * a0), (ex + (end[0] - ex) * a1, ey + (end[1] - ey) * a1)],
             sc, 220 * (1 - a0) ** 1.3, 3 * k)
      op = max(0.08, eo * (1 - bl))
      ellipse(dr, tr, ex, ey, 13 * k, 9 * op * k, outline=sc + (oa,), width=max(1, int(tr.s(ow * k))))
      ellipse(dr, tr, ex, ey, 3 * k, 3 * k, fill=WHITE + (da,))
    nose = np.array([0, 40, surf(0, 40) - 20.]); O = self.proj(R @ nose, C)
    aa, aw_ = (165, 2.6) if self.subtle else (235, 3.2)
    for v, c in [((-45, 0, 0), (255, 90, 90)), ((0, -45, 0), (110, 230, 110)), ((0, 0, -70), CYAN)]:
      p = self.proj(R @ (nose + np.array(v, float)), C)
      line(dr, tr, [tuple(O), tuple(p)], c, aa, aw_ * k)
      ellipse(dr, tr, p[0], p[1], 3.5 * k, 3.5 * k, fill=c + (255,))

  # ------------------------------------------------------------------ HUD layer
  def gauge(self, dr, cx, cy, r, aw, col):
    tr, F = self.hud, self.fonts
    box = [tr.x(cx - r), tr.y(cy - r), tr.x(cx + r), tr.y(cy + r)]
    dr.arc(box, 135, 405, fill=(255, 255, 255, 40), width=int(tr.s(16)))
    if aw > 0.004:
      dr.arc(box, 135, 135 + 270 * aw, fill=col + (255,), width=int(tr.s(16)))
    for th, lab in [(AWARE_PRE, "pre"), (AWARE_PROMPT, "prompt")]:
      ang = math.radians(135 + 270 * th)
      line(dr, tr, [(cx + (r - 14) * math.cos(ang), cy + (r - 14) * math.sin(ang)), (cx + (r + 14) * math.cos(ang), cy + (r + 14) * math.sin(ang))], WHITE, 200, 2.5)
      text(dr, tr, (cx + (r + 30) * math.cos(ang), cy + (r + 30) * math.sin(ang)), lab, F["xs"], WHITE, 170, "mm")
    text(dr, tr, (cx, cy - 8), f"{aw * 100:.0f}%", F["big"], WHITE, 255, "mm")
    text(dr, tr, (cx, cy + 42), "AWARENESS", F["lab"], WHITE, 170, "mm")

  def bar(self, dr, x, y, w, label, v, col):
    tr, F = self.hud, self.fonts
    text(dr, tr, (x, y), label, F["sm"], WHITE, 205, "lm"); text(dr, tr, (x + w, y), f"{v * 100:.0f}%", F["sm"], WHITE, 235, "rm")
    rrect(dr, tr, (x, y + 14, x + w, y + 22), 4, fill=(255, 255, 255, 35))
    if v > 0.01:
      rrect(dr, tr, (x, y + 14, x + max(8, w * v), y + 22), 4, fill=col + (235,))

  def draw_hud(self, dr, st, t):
    tr, F, r = self.hud, self.fonts, self.r
    sc = self.state_color(st)
    aA = r.alert_strength(t, "any"); aP = r.alert_strength(t, "prompt")
    if aA < 0.02:
      panel(dr, tr, (276, 46, 876, 122), r=38)
      ellipse(dr, tr, 324, 84, 8, 8, fill=sc + (255,))
      text(dr, tr, (346, 84), "Driver Monitoring", F["h2"], WHITE, 235, "lm")
      text(dr, tr, (836, 84), "Distracted" if st["dist"] else "Attentive", F["h2"], sc, 255, "rm")
    else:
      fill = mix((22, 24, 28), ORANGE, aP); hgt = 76 + 56 * aP
      panel(dr, tr, (176, 46, 976, 46 + hgt), r=30, fill=fill + (int(150 + 90 * aA),), outline=(255, 255, 255, int(40 * aA)))
      if aP < 0.5:
        rrect(dr, tr, (176, 46, 186, 46 + hgt), 5, fill=AMBER + (int(255 * aA),))
      text(dr, tr, (576, 46 + 38 + 4 * aP), st["a1"] or "Pay Attention", F["h1"], WHITE, 255 * aA, "mm")
      if aP > 0.02:
        text(dr, tr, (576, 46 + 38 + 50), st["a2"] or "Driver Distracted", F["h2"], WHITE, 255 * aP, "mm")
    panel(dr, tr, (276, 192, 876, 236), r=22, fill=(14, 16, 20, 140))
    text(dr, tr, (300, 214), f"t = {t:6.2f} s", F["sm"], WHITE, 230, "lm")
    text(dr, tr, (576, 214), f"{st['v']:4.1f} {'mph' if self.mph else 'km/h'}", F["sm"], WHITE, 230, "mm")
    text(dr, tr, (852, 214), f"steer {st['st']:+.0f}\u00b0", F["sm"], WHITE, 230, "rm")
    prog = (t - self.start) / (self.end - self.start)
    rrect(dr, tr, (300, 230, 300 + 552 * prog, 233), 2, fill=(255, 255, 255, 120))
    panel(dr, tr, (32, 480, 402, 1210))
    aw = st["aw"]; gc = GREEN if aw > AWARE_PRE else (AMBER if aw > AWARE_PROMPT else ORANGE)
    self.gauge(dr, 217, 640, 112, aw, gc)
    y = 800
    text(dr, tr, (60, y), "DISTRACTED", F["lab"], WHITE, 170, "lm")
    pc = AMBER if st["dist"] else GREEN
    rrect(dr, tr, (270, y - 16, 374, y + 16), 16, fill=pc + (60,), outline=pc + (255,), width=max(1, int(tr.s(2))))
    text(dr, tr, (322, y), "YES" if st["dist"] else "NO", F["lab"], pc, 255, "mm")
    y, x = 850, 60
    for bit, lab in [(1, "POSE"), (2, "BLINK"), (4, "PHONE")]:
      on = bool(st["dt"] & bit); c = AMBER if on else (255, 255, 255)
      rrect(dr, tr, (x, y - 16, x + 96, y + 16), 10, fill=c + (70 if on else 18,), outline=c + (255 if on else 60,), width=max(1, int(tr.s(1.5))))
      text(dr, tr, (x + 48, y), lab, F["lab"], c, 255 if on else 120, "mm"); x += 108
    y = 905
    for lab, v, c in [("Face detected", st["face"], GREEN), ("Left eye open", st["le"], CYAN), ("Right eye open", st["re"], CYAN),
                      ("Left blink", st["lb"], VIOLET), ("Right blink", st["rb"], VIOLET), ("Sunglasses", st["sun"], AMBER), ("Phone", st["ph"], AMBER)]:
      self.bar(dr, 60, y, 314, lab, v, c); y += 42
    panel(dr, tr, (32, 1370, 1120, 1696))
    text(dr, tr, (60, 1400), "HEAD POSE vs. straight ahead", F["b"], WHITE, 235, "lm")
    text(dr, tr, (60, 1428), "last 6 s  \u00b7  + yaw = driver\u2019s right  \u00b7  \u2212 pitch = looking down", F["xs"], WHITE, 150, "lm")
    gx0, gx1, gy0, gy1, rng = 60, 820, 1452, 1676, 80.
    Y = lambda v: (gy0 + gy1) / 2 - v / rng * (gy1 - gy0) / 2
    dr.rectangle([tr.x(gx0), tr.y(Y(rng)), tr.x(gx1), tr.y(Y(YAW_TH[0]))], fill=RED + (26,))
    dr.rectangle([tr.x(gx0), tr.y(Y(-YAW_TH[0])), tr.x(gx1), tr.y(Y(-rng))], fill=RED + (26,))
    for v in (YAW_TH[0], -YAW_TH[0]):
      line(dr, tr, [(gx0, Y(v)), (gx1, Y(v))], RED, 110, 1.2)
    line(dr, tr, [(gx0, Y(-PITCH_TH)), (gx1, Y(-PITCH_TH))], VIOLET, 110, 1.2)
    for v in (-60, -30, 0, 30, 60):
      line(dr, tr, [(gx0, Y(v)), (gx1, Y(v))], WHITE, 30 if v else 70, 1)
      text(dr, tr, (gx1 + 8, Y(v)), f"{v:+d}\u00b0" if v else "0\u00b0", F["xs"], WHITE, 120, "lm")
    text(dr, tr, (gx0 + 6, Y(YAW_TH[0]) - 12), "yaw limit", F["xs"], RED, 190, "lm")
    ts = np.linspace(t - 6, t, 180)
    for arr, c in [(r.pitch_err, VIOLET), (r.yaw_err, CYAN)]:
      vals = np.interp(ts, r.tds, arr)
      pts = [(gx0 + (gx1 - gx0) * (i / 179), Y(np.clip(v, -rng, rng))) for i, v in enumerate(vals)]
      line(dr, tr, pts, c, 240, 3)
      ellipse(dr, tr, pts[-1][0], pts[-1][1], 5, 5, fill=c + (255,))
    yy = 1415
    for lab, v, c, u in [("YAW", st["ye"], CYAN, "right" if st["ye"] >= 0 else "left"),
                         ("PITCH", st["pe"], VIOLET, "up" if st["pe"] >= 0 else "down"), ("ROLL", st["rn"], (210, 210, 210), "")]:
      text(dr, tr, (900, yy), lab, F["lab"], c, 230, "lm")
      text(dr, tr, (900, yy + 34), f"{v:+.0f}\u00b0", F["num"], WHITE, 255, "lm")
      if u:
        text(dr, tr, (1100, yy + 38), u, F["xs"], WHITE, 150, "rm")
      yy += 88

  def frame(self, fi):
    t = self.start + fi / FPS
    st = self.r.state(t, self.mph)
    if self.head is not None:
      bg, hp = self.head.frame(t, fi)
    else:
      bg, hp = self.base, {}
    ov = Image.new("RGBA", (self.W * SS, self.H * SS), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov, "RGBA")
    self.draw_face(dr, st, hp)
    self.draw_hud(dr, st, t)
    ov = ov.resize((self.W, self.H), Image.LANCZOS)
    im = bg.convert("RGBA"); im.alpha_composite(ov)
    return np.asarray(im.convert("RGB"))


def preview(image_path, cfg, out_path):
  """Photo with the face anchor, face box, protected head zone and wireframe drawn on it (for `drivermover calibrate`)."""
  rd = Renderer(None, image_path, cfg, 0.0, 1.0)
  g = rd.g
  st = dict(yn=0.0, pn=0.0, rn=0.0, face=1.0, fdet=True, le=1.0, re=1.0, lb=0.0, rb=0.0, a1="", a2="", dist=False)
  ov = Image.new("RGBA", (rd.W * SS, rd.H * SS), (0, 0, 0, 0)); dr = ImageDraw.Draw(ov, "RGBA"); tr = rd.face
  rd.draw_face(dr, st, {})
  dr.polygon([tr.p(q) for q in g.allow_poly], outline=(255, 80, 200, 230), width=int(tr.s(2)))
  X0, Y0, cw, ch = g.crop
  dr.rectangle([tr.x(X0), tr.y(Y0), tr.x(X0 + cw), tr.y(Y0 + ch)], outline=(255, 255, 0, 160), width=int(tr.s(1.5)))
  r = 14 * g.k
  line(dr, tr, [(g.ex - r, g.ey), (g.ex + r, g.ey)], (255, 255, 0), 255, 2); line(dr, tr, [(g.ex, g.ey - r), (g.ex, g.ey + r)], (255, 255, 0), 255, 2)
  for sx in (-1, 1):
    ellipse(dr, tr, g.ex + sx * 28 * g.k, g.ey, 4, 4, outline=(255, 255, 0, 255), width=int(tr.s(1.5)))
  f = Fonts(max(0.6, rd.W / REF_W))
  for i, (c, t_) in enumerate([((255, 255, 0), "yellow cross = eye_mid, circles = eye centres (interocular_px apart)"),
                               ((51, 209, 122), "green brackets/wireframe = face box and head model at apparent_yaw/pitch"),
                               ((255, 80, 200), "pink outline = head/neck zone realistic-head mode may change"),
                               ((255, 255, 0), "thin yellow box = head crop used for keyframe heads")]):
    text(dr, T(max(0.6, rd.W / REF_W)), (20, 20 + 26 * i), t_, f["sm"], c, 255, "la")
  ov = ov.resize((rd.W, rd.H), Image.LANCZOS)
  im = rd.photo.convert("RGBA"); im.alpha_composite(ov); im.convert("RGB").save(out_path)
  return out_path


_R = None
def _work(fi): return _R.frame(fi)


def encode(renderer, out_path, workers=None, crf=19, progress=True):
  global _R
  if shutil.which("ffmpeg") is None:
    raise RuntimeError("ffmpeg not found - install it with:  sudo apt install ffmpeg  (macOS: brew install ffmpeg)")
  out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
  W, H = renderer.W - renderer.W % 2, renderer.H - renderer.H % 2
  ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{renderer.W}x{renderer.H}",
                         "-r", str(FPS), "-i", "-", "-vf", f"crop={W}:{H}:0:0", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                         "-pix_fmt", "yuv420p", "-map_metadata", "-1", "-fflags", "+bitexact", "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
  _R = renderer
  ctx = mp.get_context("fork")
  with ctx.Pool(workers or os.cpu_count()) as pool:
    for i, fr in enumerate(pool.imap(_work, range(renderer.nf), chunksize=4)):
      ff.stdin.write(fr.tobytes())
      if progress and (i % 30 == 0 or i == renderer.nf - 1):
        print(f"\r  rendering frame {i + 1}/{renderer.nf}", end="", file=sys.stderr, flush=True)
  ff.stdin.close(); ff.wait()
  if progress:
    print(file=sys.stderr)
  if ff.returncode:
    raise RuntimeError("ffmpeg failed")
  return out_path
