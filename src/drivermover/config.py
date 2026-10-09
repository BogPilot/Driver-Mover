"""Photo-specific settings (face anchor etc.) live in a small JSON file next to the photo."""
import json
import re
from pathlib import Path

DEFAULTS = {
  "_help": "Pixel coordinates are in the photo's own pixels (0,0 = top-left). See README 'Using your own photo'.",
  "face": {
    "eye_mid": None,             # [x, y] point between the two eyes
    "interocular_px": None,      # distance between the eye centres in pixels
    "apparent_yaw_deg": 12.0,    # how the photo's head is already turned (+ = toward image-right) for the wireframe
    "apparent_pitch_deg": 0.0,
  },
  "data_mapping": {
    "_help": "How logged head angles map onto the photo (driver-camera frame). Leave as-is for a comma 3/3X driver camera.",
    "positive_yaw_is_image_right": True,
    "positive_pitch_is_up": True
  },
  "hud": {"speed_units": "mph", "vignette": True},
  "realhead": {
    "dissolve_s": 0.2, "min_hold_s": 0.35, "hysteresis_deg": 8.0,
    "near_photo_yaw_deg": 8.0, "dedupe_yaw_deg": 5.0, "dedupe_pitch_deg": 4.0, "down_keyframe_pitch_deg": 5.0,
    "head_down_start_deg": 8.0, "head_down_full_deg": 22.0,
    "micro_shift_px": 4.0, "micro_roll_deg": 2.0,
    "allow_polygon_rel": [[-168, -162], [97, -162], [97, 8], [44, 98], [30, 108], [24, 146], [-80, 146],
                          [-86, 108], [-148, 98], [-168, 48]],
    "_allow_help": "Head/neck area that may change, as offsets from eye_mid in 'reference pixels' (scaled by interocular_px/56). Everything outside stays the original photo."
  }
}


def _merge(a, b):
  out = dict(a)
  for k, v in (b or {}).items():
    out[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
  return out


def load(path=None, image=None):
  cfg = DEFAULTS
  if path is None and image is not None:
    cand = Path(image).with_suffix(".json")
    path = cand if cand.exists() else None
  if path:
    cfg = _merge(DEFAULTS, json.loads(Path(path).read_text()))
    cfg["_path"] = str(path)
  return cfg


def save(cfg, path):
  clean = {k: v for k, v in cfg.items() if not k.startswith("_path")}
  text = json.dumps(clean, indent=2)
  # keep short number lists ([x, y] pairs) on one line so the file stays readable
  text = re.sub(r"\[\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\]", r"[\1, \2]", text)
  Path(path).write_text(text + "\n")


class Geometry:
  """Everything derived from the face anchor. Reference values were tuned on the 1152x1728 mannequin photo
  (eye_mid 808,592; interocular 56 px)."""
  REF_IOD = 56.0

  def __init__(self, cfg, image_size):
    f = cfg["face"]
    if not f.get("eye_mid") or not f.get("interocular_px"):
      raise ValueError("face.eye_mid / face.interocular_px are not set - run `drivermover calibrate --image <photo> --auto` first")
    self.W, self.H = image_size
    self.ex, self.ey = map(float, f["eye_mid"])
    self.k = float(f["interocular_px"]) / self.REF_IOD
    self.apparent_yaw = float(f["apparent_yaw_deg"]); self.apparent_pitch = float(f["apparent_pitch_deg"])
    k, ex, ey = self.k, self.ex, self.ey
    self.face_box = (ex - 108 * k, ey - 122 * k, ex + 84 * k, ey + 130 * k)
    # head crop used by realistic-head mode (canonical 400x420 at k=1)
    self.crop = (int(round(ex - 221 * k)), int(round(ey - 192 * k)), int(round(400 * k)), int(round(420 * k)))
    self.allow_poly = [(ex + dx * k, ey + dy * k) for dx, dy in cfg["realhead"]["allow_polygon_rel"]]
