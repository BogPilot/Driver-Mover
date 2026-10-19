"""Wheel geometry, rotation sign, compositing guarantees and auto calibration on the bundled mannequin photo."""
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from drivermover import config as C
from drivermover.cli import EXAMPLE_DIR
from drivermover.wheel import WheelCompositor, ellipse_frame, find_wheel, rim_point, rot, screen_angle

PHOTO = EXAMPLE_DIR / "photo.jpg"
ELL = (976.8, 953.2, 331.2, 507.6, 46.5)


def test_ellipse_frame_maps_circle_to_ellipse():
  c, A = ellipse_frame(ELL)
  assert np.linalg.det(A) > 0                               # camera handedness kept
  cx, cy, w, h, ang = ELL; t = math.radians(ang)
  R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
  for q in range(0, 360, 15):
    p = rim_point(c, A, q) - [cx, cy]
    u = R @ p
    assert (u[0] / (w / 2)) ** 2 + (u[1] / (h / 2)) ** 2 == pytest.approx(1.0, abs=1e-6)
  top = rim_point(c, A, 0) - c
  assert abs(top[0]) < 1.0 and top[1] < 0                  # 12 o'clock straight up from the centre


def test_rotation_is_clockwise_on_screen():
  up = np.array([0.0, -1.0])
  assert np.allclose(rot(90) @ up, [1, 0])                  # up -> right = clockwise (y down)


def test_left_turn_is_clockwise_in_camera_view():
  # + angle = left from the driver's seat (counter-clockwise for the driver); the camera faces the driver -> clockwise
  assert screen_angle(30) == 30 and screen_angle(-90) == -90
  assert screen_angle(30, clockwise=False) == -30
  c, A = ellipse_frame(ELL)
  p = rim_point(c, A, screen_angle(45))
  assert p[0] > c[0] + 50                                   # the 12 o'clock mark moved to image-right


@pytest.fixture(scope="module")
def comp():
  cfg = C.load(None, PHOTO)
  im = np.asarray(Image.open(PHOTO).convert("RGB"))
  return im, WheelCompositor(im, cfg, warp_spokes=True)


def test_bundled_example_has_wheel_calibration():
  cfg = C.load(None, PHOTO)
  assert cfg["wheel"]["ellipse"] and len(cfg["wheel"]["ellipse"]) == 5
  assert C.load()["wheel"]["ellipse"] is None             # defaults: not calibrated


def test_nothing_outside_the_wheel_interior_changes(comp):
  im, w = comp
  c, A = w.c, w.A
  inside = np.zeros(im.shape[:2], np.uint8)
  pts = np.array([rim_point(c, A, q, w.cfg["rim_inner"] + 0.04) for q in range(0, 360, 2)], np.int32)
  cv2.fillPoly(inside, [pts], 1)
  for ang in (0.0, 37.0, -271.9, 417.6):
    out = w.apply(im, ang)
    diff = np.any(out != im, axis=2)
    assert not (diff & (inside == 0)).any(), ang            # head, rim, body, background untouched
    hands = w.hand[..., 0] > 0.999
    assert not (diff & hands).any(), ang                    # hands stay in front, pixel-exact


def test_full_turn_matches_and_motion_is_visible(comp):
  im, w = comp
  a0 = w.apply(im, 12.0).astype(int); a360 = w.apply(im, 372.0).astype(int)
  assert np.abs(a0 - a360).max() <= 2
  moved = np.abs(w.apply(im, 90.0).astype(int) - im.astype(int)).sum(axis=2) > 6
  assert moved.sum() > 2000                                 # spokes really move


def test_small_angle_is_close_to_photo(comp):
  im, w = comp
  d = np.abs(w.apply(im, 0.0).astype(int) - im.astype(int))
  assert np.percentile(d, 99.9) <= 12 and d.mean() < 0.3   # no pop when the wheel leaves centre


def test_auto_calibration_finds_example_wheel():
  cfg = C.load(None, PHOTO)
  img = cv2.imread(str(PHOTO))
  e, score = find_wheel(img, cfg)
  assert abs(e[0] - ELL[0]) < 15 and abs(e[1] - ELL[1]) < 15
  assert abs(e[2] - ELL[2]) < 0.08 * ELL[2] and abs(e[3] - ELL[3]) < 0.08 * ELL[3]
  assert abs(e[4] - ELL[4]) < 6 and score > 5


def test_default_leaves_photo_untouched():
  cfg = C.load(None, PHOTO)
  im = np.asarray(Image.open(PHOTO).convert("RGB"))
  w = WheelCompositor(im, cfg)
  assert not w.warp_spokes
  for ang in (0.0, 45.0, -271.9):
    assert np.array_equal(w.apply(im, ang), im)
  assert w.hand_overlay.max() > 0.99 and w.hand_overlay.mean() < 0.2
