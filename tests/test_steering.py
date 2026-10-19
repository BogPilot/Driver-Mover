"""Steering signals: angle extraction, who-is-steering classification, interpolation to video frames."""
import numpy as np

from drivermover.data import (WHO_DRIVER, WHO_OPENPILOT, WHO_OVERRIDE, align_prev, classify_steering, hold_flags,
                              steering_at, unwrap_angle)


def test_unwrap_keeps_multi_turn_angles():
  a = np.r_[np.linspace(0, -406.7, 300), np.linspace(-406.7, 417.6, 600)]
  assert np.allclose(unwrap_angle(a), a)


def test_unwrap_repairs_wrapped_log():
  true = np.linspace(-30, 400, 500)                       # > 1 turn left
  wrapped = (true + 180) % 360 - 180                       # as if logged in -180..180
  assert np.allclose(unwrap_angle(wrapped), true)
  assert np.allclose(unwrap_angle([5.0]), [5.0])


def test_hold_flags_debounces_pressed():
  ts = np.arange(0, 2, 0.01)
  f = np.zeros_like(ts, bool); f[50] = True                # one 10 ms blip at 0.5 s
  h = hold_flags(ts, f, 0.3)
  assert h[50] and h[79] and not h[81] and not h[49]
  assert not hold_flags(ts, np.zeros_like(f)).any()


def test_classify():
  lat = [False, False, True, True]
  pr = [False, True, False, True]
  assert list(classify_steering(lat, pr)) == [WHO_DRIVER, WHO_DRIVER, WHO_OPENPILOT, WHO_OVERRIDE]


def test_align_prev_step_signal():
  t_src = np.array([1.0, 2.0, 3.0]); v = np.array([True, False, True])
  out = align_prev(t_src, v, np.array([0.5, 1.0, 1.5, 2.99, 3.5]), False)
  assert list(out.astype(bool)) == [False, True, True, False, True]
  assert list(align_prev([], [], np.array([1.0, 2.0]), 0.0)) == [0.0, 0.0]


def test_steering_at_interpolates_between_100hz_samples():
  tcs = np.arange(0, 1, 0.01)
  ang = -300 * tcs                                         # 300 deg/s turn to the right
  cmd = ang + 5
  lat = tcs > 0.5
  who = classify_steering(lat, np.zeros_like(lat))
  fps = 30
  vals = [steering_at(i / fps, tcs, ang, cmd, lat, who) for i in range(29)]
  assert np.allclose([v["st"] for v in vals], [-300 * i / fps for i in range(29)], atol=1e-6)
  assert np.allclose([v["st_cmd"] - v["st"] for v in vals], 5)
  assert vals[10]["who"] == WHO_DRIVER and not vals[10]["lat"]
  assert vals[20]["who"] == WHO_OPENPILOT and vals[20]["lat"]
