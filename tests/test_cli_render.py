"""Config round trip, CLI flags and a full frame render with a stub route (no rlogs needed)."""
import json

import numpy as np
import pytest

from drivermover import cli
from drivermover import config as C
from drivermover.cli import EXAMPLE_DIR
from drivermover.data import WHO_OPENPILOT, WHO_OVERRIDE
from drivermover.overlay import Renderer, deg


def test_config_roundtrip(tmp_path):
  cfg = C.load(None, EXAMPLE_DIR / "photo.jpg")
  p = tmp_path / "x.json"; C.save(cfg, p)
  txt = p.read_text()
  assert '"ellipse": [976.8, 953.2, 331.2, 507.6, 46.5]' in txt
  assert C.load(p)["wheel"]["ellipse"] == cfg["wheel"]["ellipse"]
  assert json.loads(txt)["face"]["eye_mid"] == [808, 592]


def test_cli_has_no_wheel_flag(capsys):
  with pytest.raises(SystemExit):
    cli.main(["render", "--help"])
  assert "--no-wheel" in capsys.readouterr().out


def test_deg_format():
  assert deg(-0.3) == "0\u00b0" and deg(-271.9) == "-272\u00b0" and deg(4.6) == "+5\u00b0"


class StubRoute:
  tds = np.linspace(0, 30, 600)
  yaw_err = np.zeros(600); pitch_err = np.zeros(600)
  def __init__(self, who, ang, cmd, lat):
    self.s = dict(pn=0.0, yn=0.0, rn=0.0, ye=0.0, pe=0.0, face=0.98, le=0.9, re=0.9, lb=0.0, rb=0.0, sun=0.0, ph=0.0,
                  aw=1.0, dist=False, dt=0, fdet=True, a1="", a2="", ast="normal", v=20.0, st=ang, st_cmd=cmd, lat=lat, who=who)
  def baseline(self, *a): return 0.0, 0.0, 0.0
  def state(self, t, mph=True): return dict(self.s)
  def alert_strength(self, t, kind): return 0.0


@pytest.mark.parametrize("who,ang,cmd,lat", [(0, -271.9, -271.9, False), (WHO_OPENPILOT, 12.0, 14.0, True), (WHO_OVERRIDE, -8.0, 25.0, True)])
def test_render_frame_with_wheel(who, ang, cmd, lat):
  cfg = C.load(None, EXAMPLE_DIR / "photo.jpg")
  r = StubRoute(who, ang, cmd, lat)
  with_w = Renderer(r, EXAMPLE_DIR / "photo.jpg", cfg, 5.0, 1.0, wheel=True).frame(0)
  without = Renderer(r, EXAMPLE_DIR / "photo.jpg", cfg, 5.0, 1.0, wheel=False).frame(0)
  assert with_w.shape == without.shape == (1728, 1152, 3)
  d = np.any(with_w != without, axis=2)
  assert d[1180:1400, 650:1130].any()                      # HUD wheel panel drawn
  assert not d[:440, :].any()                               # top of the frame (HUD bar, head) identical
  rd = Renderer(r, EXAMPLE_DIR / "photo.jpg", cfg, 5.0, 1.0, wheel=True)
  hands = rd.wheel.hand_overlay > 0.999
  assert not d[hands].any()                                 # the rim indicator never crosses the hands
  assert d[700:1200, 780:1152].any()                        # ... but is drawn on the rim


def test_cli_has_warp_spokes_flag(capsys):
  with pytest.raises(SystemExit):
    cli.main(["render", "--help"])
  assert "--warp-spokes" in capsys.readouterr().out
