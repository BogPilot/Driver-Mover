"""Moving hands (--hands on): grips follow the rim, push-pull never lets go with both hands, the head is untouched,
the wheel never covers a hand, and the home pose reproduces the photo."""
import math

import numpy as np
import pytest
from PIL import Image

from drivermover import config as C
from drivermover.cli import EXAMPLE_DIR
from drivermover.hands import HANDS, HandRig, assets_for, plan_grips
from drivermover.overlay import Renderer
from drivermover.wheel import rim_point

PHOTO = EXAMPLE_DIR / "photo.jpg"


@pytest.fixture(scope="module")
def rig():
  cfg = C.load(None, PHOTO)
  return HandRig(np.asarray(Image.open(PHOTO).convert("RGB")), cfg, assets_for(PHOTO))


def sweeps():
  """Synthetic steering traces (screen angle, deg) at 100 Hz: fast full-lock turns, a slow drift, a wiggle."""
  t = np.arange(0, 20, 0.01)
  fast = np.interp(t, [0, 2, 3.5, 6, 7, 10, 12, 20], [0, -400, -400, 400, 380, 0, 0, 0])
  slow = 60 * np.sin(t / 3.0)
  wiggle = 25 * np.sin(2 * math.pi * 0.7 * t) + 8 * np.sin(2 * math.pi * 3.1 * t)
  jerk = np.where(t < 5, 0, np.where(t < 5.6, (t - 5) / 0.6 * -390, -390))
  return t, [fast, slow, wiggle, jerk]


def test_assets_bundled():
  assert assets_for(PHOTO) is not None
  assert assets_for(EXAMPLE_DIR / "keyframes" / "nope.jpg") is None


def test_grip_follows_rim(rig):
  for h in HANDS:
    R = rig.rig[h]
    for q in np.linspace(R["range"][0], R["range"][1], 7):
      T = rig.hand_transform(h, q)
      G0 = rig.grip_point(R["home"], h)
      assert np.allclose(T[:, :2] @ G0 + T[:, 2], rig.grip_point(q, h), atol=1e-6)     # grip lands on the rim
      assert np.allclose(rig.grip_point(q, h), rim_point(rig.c, rig.A, q, R.get("grip_r", rig.r_grip)))


def test_held_grip_turns_with_wheel(rig):
  t, traces = sweeps()
  for phi in traces:
    p = plan_grips(t, phi, rig.rig)
    for h in HANDS:
      on, q = p[h]["on"], p[h]["q"]
      lo, hi = rig.rig[h]["range"]
      both_on = on[1:] & on[:-1]
      dq, dphi = np.diff(q)[both_on], np.diff(phi)[both_on]
      inside = (q[1:][both_on] > lo + 1e-6) & (q[1:][both_on] < hi - 1e-6)
      assert np.allclose(dq[inside], dphi[inside], atol=1e-6)          # holding on = moving with the rim
      assert q.min() >= lo - 1e-6 and q.max() <= hi + 1e-6           # never outside the comfortable range
      assert np.abs(np.diff(q)).max() < 12                             # no jumps (100 Hz)


def test_shuffle_never_drops_both_hands(rig):
  t, traces = sweeps()
  for phi in traces:
    p = plan_grips(t, phi, rig.rig)
    assert not np.any(~p["right"]["on"] & ~p["left"]["on"])
  # the big turns really shuffle, and both hands end up back home when the wheel rests
  p = plan_grips(t, traces[0], rig.rig)
  regrips = sum(int(np.sum(np.diff(p[h]["on"].astype(int)) == -1)) for h in HANDS)
  assert regrips >= 6
  for h in HANDS:
    assert abs(p[h]["q"][-1] - rig.rig[h]["home"]) < 26


def test_home_pose_reproduces_photo(rig):
  x0, y0, x1, y1 = rig.roi
  roi, ha = rig.render_roi(0.0, {h: (rig.rig[h]["home"], 0.0) for h in HANDS})
  d = np.abs(roi - rig.base_roi).max(-1)
  assert d.mean() < 0.6 and np.percentile(d, 99) < 12


def test_wheel_never_covers_a_hand(rig):
  for phi, qs in [(0, (300, 92)), (-200, (262, 80)), (157, (335, 106)), (400, (300, 92)), (-90, (280, 100))]:
    g = {h: (q, 0.0) for h, q in zip(HANDS, qs)}
    img, ha = rig.render_roi(phi, g)
    for h in HANDS:
      _, hand = rig._skin(h, rig.pose(h, *g[h]))
      solid = hand[..., 3] > 0.995
      if solid.any():
        assert np.abs(img[solid] - hand[..., :3][solid]).max() < 2.0  # hand pixels are the hand, on top of everything


def test_head_zone_untouched(rig):
  cfg = C.load(None, PHOTO)
  g = C.Geometry(cfg, (1152, 1728))
  assert rig.roi[1] > max(p[1] for p in g.allow_poly)                 # rig region is below the realistic-head zone
  frame = np.asarray(Image.open(PHOTO).convert("RGB"))
  rig.set_plan(np.arange(0, 3, 0.01), np.linspace(0, -300, 300))
  out, hm = rig.apply(frame, 1.5, -150.0)
  x0, y0, x1, y1 = rig.roi
  outside = np.ones(frame.shape[:2], bool); outside[y0:y1, x0:x1] = False
  assert np.array_equal(out[outside], frame[outside])                  # nothing outside the arm region changes
  assert np.array_equal(out[:y0], frame[:y0])                          # head/face/torso top pixel-exact
  assert hm[outside].max() == 0 and hm.max() > 0.99


class _Route:
  tds = np.linspace(0, 30, 600)
  yaw_err = np.zeros(600); pitch_err = np.zeros(600)
  tcs = np.arange(0, 30, 0.01)
  steer = np.interp(tcs, [0, 5, 6, 30], [0, 0, -270, -270])
  def __init__(self):
    self.s = dict(pn=0.0, yn=0.0, rn=0.0, ye=0.0, pe=0.0, face=0.98, le=0.9, re=0.9, lb=0.0, rb=0.0, sun=0.0, ph=0.0,
                  aw=1.0, dist=False, dt=0, fdet=True, a1="", a2="", ast="normal", v=20.0, st_cmd=0.0, lat=False, who=0)
  def baseline(self, *a): return 0.0, 0.0, 0.0
  def state(self, t, mph=True): return dict(self.s, st=float(np.interp(t, self.tcs, self.steer)))
  def alert_strength(self, t, kind): return 0.0


def test_renderer_hands_on_off():
  cfg = C.load(None, PHOTO)
  r = _Route()
  on = Renderer(r, PHOTO, cfg, 5.4, 1.0, wheel=True, hands=assets_for(PHOTO))
  off = Renderer(r, PHOTO, cfg, 5.4, 1.0, wheel=True)
  a, b = on.frame(0), off.frame(0)
  d = np.any(a != b, axis=2)
  assert d[740:1260, 560:1152].mean() > 0.05                           # arms / spokes moved
  assert not d[:740].any()                                             # head, face overlay, HUD top identical
