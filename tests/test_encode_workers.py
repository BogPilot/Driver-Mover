"""Regression: a 2-worker encode must finish (macOS used to deadlock on fork)."""
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from drivermover import config as C
from drivermover.cli import EXAMPLE_DIR
from drivermover.hands import assets_for
from drivermover.overlay import Renderer, _worker_context, encode

TIMEOUT_S = 180


class _Route:
  tds = np.linspace(0, 30, 600)
  yaw_err = np.zeros(600)
  pitch_err = np.zeros(600)
  tcs = np.arange(0, 30, 0.01)
  steer = np.zeros_like(tcs)

  def __init__(self):
    self.s = dict(pn=0.0, yn=0.0, rn=0.0, ye=0.0, pe=0.0, face=0.98, le=0.9, re=0.9, lb=0.0, rb=0.0,
                  sun=0.0, ph=0.0, aw=1.0, dist=False, dt=0, fdet=True, a1="", a2="", ast="normal",
                  v=20.0, st=0.0, st_cmd=0.0, lat=False, who=0)

  def baseline(self, *a):
    return 0.0, 0.0, 0.0

  def state(self, t, mph=True):
    return dict(self.s)

  def alert_strength(self, t, kind):
    return 0.0


def test_worker_context_matches_platform():
  method = _worker_context().get_start_method()
  if sys.platform == "darwin":
    assert method == "spawn"
  else:
    assert method == "fork"


def _encode_two_workers(out):
  os.environ.setdefault("MPLBACKEND", "Agg")
  os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
  os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
  cfg = C.load(None, EXAMPLE_DIR / "photo.jpg")
  photo = EXAMPLE_DIR / "photo.jpg"
  rd = Renderer(_Route(), photo, cfg, 0.0, 0.5, wheel=True, hands=assets_for(photo))
  encode(rd, Path(out), workers=2, progress=False)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required to encode")
def test_two_worker_encode_finishes(tmp_path):
  """OpenCV is imported in this process first, which is what used to hang fork workers."""
  out = tmp_path / "two_workers.mp4"
  ctx = mp.get_context("spawn")
  proc = ctx.Process(target=_encode_two_workers, args=(str(out),))
  proc.start()
  proc.join(TIMEOUT_S)
  if proc.is_alive():
    proc.terminate()
    proc.join(5)
    pytest.fail(f"2-worker encode did not finish within {TIMEOUT_S} s")
  assert proc.exitcode == 0, f"encode subprocess exited {proc.exitcode}"
  assert out.is_file() and out.stat().st_size > 1000
