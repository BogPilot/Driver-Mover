"""Moving hands for the bundled mannequin photo (v2 `--hands on`).

The photo is split into layers once (tools/build_hands.py): a clean plate without arms or spokes, the spokes/hub,
and the two arms cut out of the photo. Per frame they are put back together, bottom to top:

  clean plate  ->  arms (upper arm + forearm, skinned to a 2-bone skeleton)  ->  spokes turned by the steering angle
  ->  rim band (static: a turning rim looks the same)  ->  hands (rigid, gripping the rim)  ->  overlays

Each hand grips the rim at a clock angle q (camera view, clockwise from 12 o'clock).  While it holds on, q turns with
the wheel.  Before it leaves its comfortable range it lets go and slides back along the rim (push-pull steering):
one hand at a time, never both; a hand that cannot let go (the other one is moving) slips on the rim at the limit.
When the wheel is still the hands drift back to their home grips one at a time.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .wheel import ellipse_frame, rim_point, screen_angle

HANDS = ("right", "left")


def smoothstep(x, a, b):
  t = np.clip((np.asarray(x, np.float32) - a) / (b - a), 0, 1)
  return t * t * (3 - 2 * t)


def ease(u):
  u = min(max(u, 0.0), 1.0)
  return u * u * (3 - 2 * u)


def tangent_deg(c, A, q):
  """Screen direction (deg) of the rim tangent at clock angle q (direction of increasing q)."""
  d = A @ np.array([math.cos(math.radians(q)), math.sin(math.radians(q))])
  return math.degrees(math.atan2(d[1], d[0]))


def _angdiff(a, b):
  return (a - b + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------------------------- grip scheduler
def plan_grips(ts, phi, rig, release_margin=4.0, regrip_past_home=22.0, calm_speed=12.0, calm_time=0.8, recentre_deg=12.0, min_hold=0.15):
  """Push-pull grip plan for wheel screen-angle samples phi(ts) (deg, clockwise in the photo, uniform ts).

  Returns dict of arrays per hand: q (grip clock angle), lift (0..1, 1 = hand off the rim mid-slide), on (bool)."""
  ts = np.asarray(ts, float); phi = np.asarray(phi, float); n = len(ts)
  dt = float(ts[1] - ts[0]) if n > 1 else 0.01
  om = np.gradient(phi, dt) if n > 1 else np.zeros(n)
  k = max(1, int(round(0.08 / dt)))
  om = np.convolve(om, np.ones(k) / k, mode="same")
  P = {h: dict(home=float(rig[h]["home"]), lo=float(rig[h]["range"][0]), hi=float(rig[h]["range"][1]),
               wmin=float(rig[h].get("regrip_min_speed", 3.0))) for h in HANDS}
  st = {h: dict(on=True, b=P[h]["home"] - phi[0], t_on=-1e9, frm=0.0, to=0.0, t0=0.0, T=1.0) for h in HANDS}
  out = {h: dict(q=np.zeros(n), lift=np.zeros(n), on=np.ones(n, bool)) for h in HANDS}
  calm = 0.0
  for i in range(n):
    t, f, w = ts[i], phi[i], om[i]
    calm = calm + dt if abs(w) < calm_speed else 0.0
    # finish slides
    for h in HANDS:
      s = st[h]
      if not s["on"] and t >= s["t0"] + s["T"]:
        s.update(on=True, b=s["to"] - f, t_on=t)
    # let go before a hand leaves its range (only if the other hand holds on)
    for h in HANDS:
      s, o, p = st[h], st[HANDS[1 - HANDS.index(h)]], P[h]
      if not s["on"] or not o["on"] or t - o["t_on"] < min_hold or t - s["t_on"] < min_hold:
        continue
      q = min(max(f + s["b"], p["lo"]), p["hi"])
      to = None
      wmin = p["wmin"]
      if w > wmin and q > p["hi"] - release_margin:
        to = p["home"] - regrip_past_home
      elif w < -wmin and q < p["lo"] + release_margin:
        to = p["home"] + regrip_past_home
      elif calm >= calm_time and abs(q - p["home"]) > recentre_deg and all(abs(f + st[x]["b"] - P[x]["home"]) <= abs(q - p["home"]) for x in HANDS if st[x]["on"]):
        to = p["home"]
      if to is not None:
        to = min(max(to, p["lo"] + 2), p["hi"] - 2)
        s.update(on=False, frm=q, to=to, t0=t, T=float(np.clip(0.24 + abs(to - q) / 420.0, 0.28, 0.5)))
        if calm >= calm_time:
          calm = -0.3          # one hand at a time when re-centring
    for h in HANDS:
      s, p = st[h], P[h]
      if s["on"]:
        q = f + s["b"]
        if q < p["lo"] or q > p["hi"]:          # cannot let go right now: the hand slips on the rim
          q = min(max(q, p["lo"]), p["hi"]); s["b"] = q - f
        out[h]["q"][i] = q
      else:
        u = (t - s["t0"]) / s["T"]
        out[h]["q"][i] = s["frm"] + (s["to"] - s["frm"]) * ease(u)
        out[h]["lift"][i] = math.sin(math.pi * min(max(u, 0.0), 1.0))
        out[h]["on"][i] = False
  return out


# ---------------------------------------------------------------------------------------------- the rig
class HandRig:
  def __init__(self, base_rgb, cfg, assets_dir, vig=None, plan_times=None, plan_phi=None):
    d = Path(assets_dir)
    self.rig = json.loads((d / "rig.json").read_text())
    self.c, self.A = ellipse_frame(cfg["wheel"]["ellipse"], cfg["wheel"].get("up_dir", [0, -1]))
    self.cw = bool(cfg["wheel"].get("positive_angle_is_clockwise_in_image", True))
    self.r_grip = float(self.rig.get("grip_r", 0.9))
    x0, y0, x1, y1 = self.roi = tuple(int(v) for v in self.rig["roi"])
    H, W = base_rgb.shape[:2]
    if x1 > W or y1 > H:
      raise ValueError("hands assets do not match this photo")
    rd = lambda n: cv2.cvtColor(cv2.imread(str(d / n), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA if n != "bg.png" else cv2.COLOR_BGR2RGB)
    v = (vig[y0:y1, x0:x1].astype(np.float32) if vig is not None else np.ones((y1 - y0, x1 - x0, 1), np.float32))
    self.bg = rd("bg.png").astype(np.float32) * v
    self.base_roi = base_rgb[y0:y1, x0:x1].astype(np.float32)

    def premul(rgba):
      a = rgba[..., 3:4].astype(np.float32) / 255.0
      return np.dstack([rgba[..., :3].astype(np.float32) * v * a, a])
    self.spokes = premul(rd("spokes.png"))
    self.arm = {"right": premul(rd("armR.png")), "left": premul(rd("armL.png"))}
    # a sprite cut by the photo edge fades out towards that edge once it moves away from it (no hard cut line)
    edge = (1 - smoothstep(np.arange(x0, x1, dtype=np.float32), W - 14.0, W - 1.0))[None, :].repeat(y1 - y0, 0)
    self.edge_fade = edge if x1 >= W else np.ones_like(edge)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    self.xs, self.ys = xs, ys
    # wheel-plane coordinates of each ROI pixel -> static rim band (in front of the arms, behind the hands)
    Ai = np.linalg.inv(self.A)
    u = Ai[0, 0] * (xs - self.c[0]) + Ai[0, 1] * (ys - self.c[1]); w_ = Ai[1, 0] * (xs - self.c[0]) + Ai[1, 1] * (ys - self.c[1])
    rr = np.hypot(u, w_)
    rin = float(cfg["wheel"].get("rim_inner", 0.8))
    self.rim_a = (smoothstep(rr, rin - 0.025, rin - 0.005) * (1 - smoothstep(rr, 1.0, 1.02)))[..., None]
    # source skinning: per arm the weights of the upper-arm and forearm bones (linear-blend skinning) and the split
    # between the arm (behind the wheel) and the hand (in front, rigid). The forearm runs on under the hand a little
    # so a bent wrist never opens a gap.
    self.bones, self.hand_w, self.arm_w = {}, {}, {}
    for h in HANDS:
      R = self.rig[h]
      dw = self._side(np.array(R["wrist_line"], np.float32), toward=self._grip0(h))
      self.hand_w[h] = smoothstep(dw, -3.0, 9.0)
      self.arm_w[h] = 1 - smoothstep(dw, 9.0, 18.0)
      if "shoulder" in R:
        de = self._side(np.array(R["elbow_line"], np.float32), toward=np.array(R["elbow"], np.float32) + [0.0, 40.0])
        w_up = 1 - smoothstep(de, -14.0, 14.0)
      else:
        w_up = np.zeros_like(dw)
      self.bones[h] = np.dstack([w_up, 1 - w_up]).astype(np.float32)
    self.plan = None
    if plan_times is not None:
      self.set_plan(plan_times, plan_phi)

  # ------------------------------------------------------------------ geometry helpers
  def _side(self, line, toward):
    """Signed distance (px) of each ROI pixel from the infinite line, positive on the side of point `toward`."""
    (ax, ay), (bx, by) = line
    nx, ny = -(by - ay), (bx - ax); L = math.hypot(nx, ny); nx, ny = nx / L, ny / L
    if (toward[0] - ax) * nx + (toward[1] - ay) * ny < 0:
      nx, ny = -nx, -ny
    return (self.xs - ax) * nx + (self.ys - ay) * ny

  def _grip0(self, h):
    return self.grip_point(self.rig[h]["home"], h)

  def grip_point(self, q, h="right"):
    return rim_point(self.c, self.A, q, float(self.rig[h].get("grip_r", self.r_grip)))

  def hand_transform(self, h, q, lift=0.0):
    """2x3 similarity: photo hand pose (grip at home) -> grip at clock angle q, rotated with the rim tangent."""
    q0 = self.rig[h]["home"]
    G0, G = self._grip0(h), self.grip_point(q, h)
    rot = 0.8 * _angdiff(tangent_deg(self.c, self.A, q), tangent_deg(self.c, self.A, q0))
    sp = lambda qq: np.linalg.norm(self.A @ np.array([math.cos(math.radians(qq)), math.sin(math.radians(qq))]))
    s = float(np.clip((sp(q) / sp(q0)) ** 0.5, 0.92, 1.08)) * (1 + 0.035 * lift)
    out = (G - self.c); out = out / (np.linalg.norm(out) + 1e-9)
    G = G + out * 9.0 * lift
    a = math.radians(rot); R = s * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return np.hstack([R, (G - R @ G0)[:, None]])

  @staticmethod
  def bone_transform(A0, B0, A1, B1, width_gain=0.0):
    """2x3 affine taking segment A0B0 to A1B1: rotation + stretch along the bone, width kept (x stretch**width_gain)."""
    A0, B0, A1, B1 = (np.asarray(p, float) for p in (A0, B0, A1, B1))
    d0, d1 = B0 - A0, B1 - A1; L0, L1 = np.linalg.norm(d0), np.linalg.norm(d1)
    u0, u1 = d0 / L0, d1 / L1; n0, n1 = np.array([-u0[1], u0[0]]), np.array([-u1[1], u1[0]])
    s = L1 / L0; wscale = s ** width_gain
    M = np.outer(u1 * s, u0) + np.outer(n1 * wscale, n0)
    return np.hstack([M, (A1 - M @ A0)[:, None]])

  def pose(self, h, q, lift=0.0):
    """Bone transforms [upper, fore, hand] (2x3) for hand h gripping at q."""
    R = self.rig[h]
    Th = self.hand_transform(h, q, lift)
    W0 = np.array(R["wrist"], float); W = Th[:, :2] @ W0 + Th[:, 2]
    E0 = np.array(R["elbow"], float)
    E = E0 + float(R.get("elbow_follow", 0.0)) * (W - W0)
    if "shoulder" in R:
      S0 = np.array(R["shoulder"], float)
      Tu = self.bone_transform(S0, E0, S0, E)
    else:
      Tu = np.hstack([np.eye(2), np.zeros((2, 1))])
    Tf = self.bone_transform(E0, W0, E, W, width_gain=0.25)
    return [Tu, Tf, Th]

  # ------------------------------------------------------------------ plan
  def set_plan(self, ts, phi):
    self.plan_t = np.asarray(ts, float); self.plan_phi = np.asarray(phi, float)
    self.plan = plan_grips(self.plan_t, self.plan_phi, self.rig)

  def grips_at(self, t):
    if self.plan is None:
      return {h: (float(self.rig[h]["home"]), 0.0) for h in HANDS}
    return {h: (float(np.interp(t, self.plan_t, self.plan[h]["q"])), float(np.interp(t, self.plan_t, self.plan[h]["lift"]))) for h in HANDS}

  # ------------------------------------------------------------------ rendering
  @staticmethod
  def _inv_maps(T, X, Y):
    Ti = np.linalg.inv(np.vstack([T, [0, 0, 1]]))[:2]
    return Ti[0, 0] * X + Ti[0, 1] * Y + Ti[0, 2], Ti[1, 0] * X + Ti[1, 1] * Y + Ti[1, 2]

  def _skin(self, h, Ts):
    """Warp arm h for bone transforms [upper, fore, hand]: arm = inverse linear-blend skinning of the upper-arm and
    forearm bones (fixed-point iteration), hand = rigid. Returns premultiplied (arm, hand) sprites over the ROI."""
    X, Y = self.xs, self.ys
    x0, y0 = self.roi[:2]
    (ux, uy), (fx, fy) = self._inv_maps(Ts[0], X, Y), self._inv_maps(Ts[1], X, Y)
    sx, sy = fx, fy
    for _ in range(4):
      w = cv2.remap(self.bones[h], (sx - x0).astype(np.float32), (sy - y0).astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
      w0 = w[..., 0] / np.maximum(w.sum(-1), 1e-6)
      sx, sy = w0 * ux + (1 - w0) * fx, w0 * uy + (1 - w0) * fy
    mx, my = (sx - x0).astype(np.float32), (sy - y0).astype(np.float32)
    arm = cv2.remap(self.arm[h], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    arm *= cv2.remap(self.arm_w[h], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., None]
    hx, hy = self._inv_maps(Ts[2], X, Y)
    hx, hy = (hx - x0).astype(np.float32), (hy - y0).astype(np.float32)
    hand = cv2.remap(self.arm[h], hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    hw = cv2.remap(self.hand_w[h], hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # how far the photo-edge cut of the sprite moved into the frame (0 = still on the edge)
    Wf = float(self.edge_fade.shape[1] + x0)
    ys_ = np.nonzero(self.arm[h][:, -1, 3] > 0.1)[0]
    moved = 0.0
    if self.roi[2] >= Wf and len(ys_):
      ym = float(ys_.mean() + y0)
      moved = float(np.clip(((Wf - 1) - (Ts[2][0, 0] * (Wf - 1) + Ts[2][0, 1] * ym + Ts[2][0, 2])) / 8.0, 0, 1))
    if moved > 0:
      ef = cv2.remap(self.edge_fade, hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
      hw = hw * (1 - moved * (1 - ef))
      af = cv2.remap(self.edge_fade, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
      arm *= (1 - moved * (1 - af))[..., None]
    hand *= hw[..., None]
    return arm, hand

  def wheel_map(self, phi):
    """Inverse map that turns the wheel plane by phi (deg, clockwise in the photo) about the hub."""
    a = math.radians(phi)
    Rm = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])   # inverse rotation in (sin, -cos) coords
    Mi = self.A @ Rm @ np.linalg.inv(self.A)
    dx, dy = self.xs - self.c[0], self.ys - self.c[1]
    x0, y0 = self.roi[:2]
    return ((Mi[0, 0] * dx + Mi[0, 1] * dy + self.c[0] - x0).astype(np.float32),
            (Mi[1, 0] * dx + Mi[1, 1] * dy + self.c[1] - y0).astype(np.float32))

  def render_roi(self, phi, grips):
    """Composite the arm region. Returns (rgb float ROI, hand alpha ROI)."""
    over = lambda dst, src, a: dst * (1 - a) + src
    img = self.bg.copy()
    hands = []
    for h in HANDS:
      q, lift = grips[h]
      arm, hand = self._skin(h, self.pose(h, q, lift))
      img = over(img, arm[..., :3], arm[..., 3:4])
      hands.append(hand)
    mx, my = self.wheel_map(phi)
    sp = cv2.remap(self.spokes, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = over(img, sp[..., :3], sp[..., 3:4])
    img = img * (1 - self.rim_a) + self.bg * self.rim_a
    ha = np.zeros(img.shape[:2], np.float32)
    for hs in hands:
      img = over(img, hs[..., :3], hs[..., 3:4]); ha = ha + hs[..., 3] * (1 - ha)
    return img, ha

  def apply(self, frame_rgb, t, phi):
    """Frame (HxWx3 uint8) with the arm region re-composited for time t / wheel angle phi; also the hand alpha (HxW)."""
    x0, y0, x1, y1 = self.roi
    roi, ha = self.render_roi(phi, self.grips_at(t))
    out = np.array(frame_rgb, copy=True)
    out[y0:y1, x0:x1] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)
    hm = np.zeros(frame_rgb.shape[:2], np.float32); hm[y0:y1, x0:x1] = ha
    return out, hm


def assets_for(image_path):
  """Bundled hands assets for the example photo, or None."""
  p = Path(image_path).resolve()
  d = Path(__file__).resolve().parent / "examples" / "mannequin"
  if p == (d / "photo.jpg").resolve() and (d / "hands" / "rig.json").exists():
    return d / "hands"
  if (p.parent / "hands" / "rig.json").exists():
    return p.parent / "hands"
  return None
