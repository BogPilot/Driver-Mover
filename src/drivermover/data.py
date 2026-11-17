"""Extract driver-monitoring signals from a route and convert them like openpilot's dmonitoringd does."""
import math
import sys

import numpy as np

from .logreader import find_segments, iter_events
from .schema_packs import select_schema, uses_v111_dm

# openpilot selfdrive/monitoring/helpers.py constants (0.9.7–0.11.0)
DT_DMON = 0.05
POSE_OFFSET_MIN_COUNT = int(60 / DT_DMON)
PITCH_NATURAL_OFFSET, YAW_NATURAL_OFFSET = 0.029, 0.097
# 0.11.1+ policy.py (natural offsets only; min/max clip bounds stayed the same)
PITCH_NATURAL_OFFSET_V111, YAW_NATURAL_OFFSET_V111 = 0.011, 0.075
PITCH_MIN_OFFSET, PITCH_MAX_OFFSET = -0.0881, 0.124
YAW_MIN_OFFSET, YAW_MAX_OFFSET = -0.0246, 0.289
YAW_TH = (math.degrees(0.4020), math.degrees(0.5042))   # strict, slack
PITCH_TH = math.degrees(0.449)                           # natural (uncalibrated) head-down limit
AWARE_PRE, AWARE_PROMPT = 8 / 11, 6 / 11
# face_orientation_from_net: model output refers to the undistorted 1928x1208 driver image, EFL 598 px
# (still the tici/tizi/mici cabin-undistorted reference in 0.11.2 policy.py)
CAM_W, CAM_H, EFL = 1928, 1208, 598.0

# All aliases across 0.9.7 .. current. Unknown union members are skipped by the reader.
WANTED = {
    "driverStateV2", "driverState", "driverStateDEPRECATED",
    "driverMonitoringState", "driverMonitoringStateDEPRECATED",
    "controlsState", "selfdriveState",
    "carState", "carControl",
    "liveCalibration", "extrinsicsCalibration",
}

# who is steering (per carState sample)
WHO_DRIVER, WHO_OPENPILOT, WHO_OVERRIDE = 0, 1, 2
WHO_LABEL = {WHO_DRIVER: "DRIVER", WHO_OPENPILOT: "OPENPILOT", WHO_OVERRIDE: "DRIVER OVERRIDE"}
PRESSED_HOLD_S = 0.3     # steeringPressed flickers at the torque threshold; hold it this long

# distractedType bits used when packing the 0.11.1+ DistractedTypes struct
DISTRACT_POSE, DISTRACT_EYE, DISTRACT_PHONE, DISTRACT_SLEEP = 1, 2, 4, 8


def unwrap_angle(deg, jump=180.0):
  """Steering angles are continuous multi-turn values (e.g. -406 deg). If a log ever wraps them into +-180 / 0..360,
  undo it: a real wheel cannot move more than `jump` degrees between two 100 Hz samples."""
  a = np.asarray(deg, float)
  if a.size < 2:
    return a.copy()
  d = np.diff(a)
  fix = np.where(d > jump, -360.0, np.where(d < -jump, 360.0, 0.0))
  return np.r_[a[0], a[0] + np.cumsum(d + fix)]


def hold_flags(ts, flags, hold_s=PRESSED_HOLD_S):
  """Keep a boolean on for `hold_s` seconds after each True sample (debounce)."""
  ts = np.asarray(ts, float); f = np.asarray(flags, bool)
  if not f.any():
    return f.copy()
  last = np.where(f, ts, -np.inf)
  last = np.maximum.accumulate(last)
  return (ts - last) <= hold_s


def classify_steering(lat_active, pressed):
  """openpilot steering (lateral control active, driver not pushing), driver override (lat active and the driver is
  applying torque), or driver (lat not active). Vectorised."""
  lat = np.asarray(lat_active, bool); pr = np.asarray(pressed, bool)
  return np.where(lat, np.where(pr, WHO_OVERRIDE, WHO_OPENPILOT), WHO_DRIVER).astype(int)


def align_prev(t_src, values, t_dst, default=0.0):
  """Most recent value of a step signal (sampled at t_src) at each t_dst."""
  t_src = np.asarray(t_src, float); v = np.asarray(values)
  if t_src.size == 0:
    return np.full(np.shape(t_dst), default, dtype=float)
  i = np.searchsorted(t_src, t_dst, "right") - 1
  out = v[np.clip(i, 0, len(v) - 1)]
  return np.where(i >= 0, out, default)


def steering_at(t, tcs, angle, cmd, lat, who):
  """Wheel state at time t (video frames fall between 100 Hz samples): angle/command interpolated, flags from the
  latest sample."""
  i = max(0, int(np.searchsorted(tcs, t, "right")) - 1)
  return dict(st=float(np.interp(t, tcs, angle)), st_cmd=float(np.interp(t, tcs, cmd)), lat=bool(lat[i]), who=int(who[i]))


def _g(obj, name, default=0.0):
  try:
    return getattr(obj, name)
  except Exception:
    return default


def _first_attr(obj, names, default=None):
  for n in names:
    try:
      return getattr(obj, n)
    except Exception:
      continue
  return default


def _event_names(m):
  """Car-event names from any generation of DriverMonitoringState."""
  for attr in ("events", "eventsDEPRECATED"):
    ev = _g(m, attr, None)
    if ev is None:
      continue
    names = []
    try:
      for x in ev:
        try:
          names.append(str(x.name))
        except Exception:
          pass
    except Exception:
      continue
    if names:
      return names
  ev = _g(_g(m, "deprecated", None), "events", None)
  if ev is not None:
    names = []
    try:
      for x in ev:
        try:
          names.append(str(x.name))
        except Exception:
          pass
    except Exception:
      names = []
    if names:
      return names
  return []


def _driver_data_row(x):
  """One side of driverStateV2 (or a synthetic V1 row) as a fixed-length list."""
  fo = list(_g(x, "faceOrientation", []) or [])
  while len(fo) < 3:
    fo.append(0.0)
  fp = list(_g(x, "facePosition", []) or [])
  while len(fp) < 2:
    fp.append(0.0)
  return (fo[:3] + fp[:2] + [
      float(_g(x, "faceProb", 0.0)),
      float(_g(x, "leftEyeProb", 0.0)),
      float(_g(x, "rightEyeProb", 0.0)),
      float(_g(x, "leftBlinkProb", 0.0)),
      float(_g(x, "rightBlinkProb", 0.0)),
      float(_g(x, "sunglassesProb", 0.0)),
      float(_g(x, "phoneProb", _g(x, "phoneUse", 0.0))),
  ])


def extract_driver_state(which, evt):
  """Return (left_row, right_row, wheel_on_right_prob) from V2 or deprecated V1."""
  if which == "driverStateV2":
    d = evt.driverStateV2
    return _driver_data_row(d.leftDriverData), _driver_data_row(d.rightDriverData), float(_g(d, "wheelOnRightProb", 0.0))
  d = _first_attr(evt, ("driverStateDEPRECATED", "driverState"))
  if d is None:
    return None
  row = _driver_data_row(d)
  # V1 is a single face. Duplicate it onto both sides; isRHD later picks the side.
  return row, row, float(_g(d, "phoneUse", 0.0)) * 0.0


def _new_dms_row(m):
  """Normalize the 0.11.1+ DriverMonitoringState tree to the pre-0.11.1 tuple."""
  v = m.visionPolicyState
  policy = str(_g(m, "activePolicy", ""))
  if "wheeltouch" in policy.lower():
    aw = float(_g(m.wheeltouchPolicyState, "awarenessPercent", 0)) / 100.0
  else:
    aw = float(_g(v, "awarenessPercent", 0)) / 100.0
  types = _g(v, "distractedTypes", None)
  dtype = 0
  if types is not None:
    if bool(_g(types, "pose", False)):
      dtype |= DISTRACT_POSE
    if bool(_g(types, "eye", False)):
      dtype |= DISTRACT_EYE
    if bool(_g(types, "phone", False)):
      dtype |= DISTRACT_PHONE
    if bool(_g(types, "sleep", False)):
      dtype |= DISTRACT_SLEEP
  pose = _g(v, "pose", None)
  pitch_off = float(_g(_g(pose, "pitchCalib", None), "offset", 0.0)) if pose is not None else 0.0
  yaw_off = float(_g(_g(pose, "yawCalib", None), "offset", 0.0)) if pose is not None else 0.0
  # Represent calibration progress on the same scale the 0.9.7 path expects
  # (valid-count vs POSE_OFFSET_MIN_COUNT). calibratedPercent is 0–100.
  def _count(calib, calibrated_flag):
    pct = float(_g(calib, "calibratedPercent", 0.0))
    if calibrated_flag and pct <= 0:
      return float(POSE_OFFSET_MIN_COUNT + 1)
    return (pct / 100.0) * POSE_OFFSET_MIN_COUNT

  calibrated = bool(_g(pose, "calibrated", False)) if pose is not None else False
  pitch_n = _count(_g(pose, "pitchCalib", None) if pose is not None else None, calibrated)
  yaw_n = _count(_g(pose, "yawCalib", None) if pose is not None else None, calibrated)
  events = []
  level = str(_g(m, "alertLevel", "none"))
  if level and level not in ("none", "AlertLevel.none"):
    events.append(level)
  return (
      events,
      bool(_g(v, "faceDetected", False)),
      bool(_g(v, "isDistracted", False)),
      int(dtype),
      aw,
      bool(_g(m, "isRHD", False)),
      pitch_off,
      pitch_n,
      yaw_off,
      yaw_n,
  )


def _old_dms_row(m):
  return (
      _event_names(m),
      bool(_g(m, "faceDetected", False)),
      bool(_g(m, "isDistracted", False)),
      int(_g(m, "distractedType", 0)),
      float(_g(m, "awarenessStatus", 0.0)),
      bool(_g(m, "isRHD", False)),
      float(_g(m, "posePitchOffset", 0.0)),
      float(_g(m, "posePitchValidCount", 0.0)),
      float(_g(m, "poseYawOffset", 0.0)),
      float(_g(m, "poseYawValidCount", 0.0)),
  )


def extract_dms(which, evt):
  """Normalize any DriverMonitoringState generation to a 10-tuple of fields."""
  if which == "driverMonitoringStateDEPRECATED":
    return _old_dms_row(evt.driverMonitoringStateDEPRECATED)
  m = evt.driverMonitoringState
  # 0.11.1+ payload has visionPolicyState; the 0.9.7–0.11.0 payload has faceDetected at the top level.
  if _g(m, "visionPolicyState", None) is not None and not hasattr(m, "faceDetected"):
    return _new_dms_row(m)
  if _g(m, "visionPolicyState", None) is not None:
    # Some schemas could theoretically have both; prefer new if the nested state is present
    # and faceDetected is only inside it.
    try:
      m.faceDetected
      return _old_dms_row(m)
    except Exception:
      return _new_dms_row(m)
  return _old_dms_row(m)


def extract_calibration_rpy(evt, which):
  src = evt.extrinsicsCalibration if which == "extrinsicsCalibration" else evt.liveCalibration
  r = list(_g(src, "rpyCalib", []) or [])
  return r


class Route:
  def __init__(self, paths, schema_dir=None, verbose=True):
    self.files = find_segments(paths)
    self.meta = select_schema(self.files, schema_dir=schema_dir, verbose=verbose)
    schema_dir = self.meta.pack_dir
    self.schema_pack = self.meta.pack
    ds, dm, cs, cc, ctl, sds, cal, seg_t = [], [], [], [], [], [], [], []
    for f in self.files:
      first = None
      for w, e in iter_events(f, schema_dir, WANTED):
        t = e.logMonoTime * 1e-9
        if w in ("driverStateV2", "driverState", "driverStateDEPRECATED"):
          row = extract_driver_state(w, e)
          if row is None:
            continue
          left, right, wor = row
          ds.append((t, left, right, wor))
          first = t if first is None else min(first, t)
        elif w in ("driverMonitoringState", "driverMonitoringStateDEPRECATED"):
          dm.append((t, *extract_dms(w, e)))
        elif w in ("controlsState", "selfdriveState"):
          c = getattr(e, w)
          row = (t, str(_g(c, "alertText1", "")), str(_g(c, "alertText2", "")), str(_g(c, "alertStatus", "normal")))
          (ctl if w == "controlsState" else sds).append(row)
        elif w == "carState":
          c = e.carState
          cs.append((t, c.vEgo, c.steeringAngleDeg, bool(_g(c, "steeringPressed", False)), float(_g(c, "steeringTorque", 0.0))))
        elif w == "carControl":
          c = e.carControl
          cc.append((t, bool(_g(c, "enabled", False)), bool(_g(c, "latActive", False)), float(_g(c.actuators, "steeringAngleDeg", 0.0))))
        elif w in ("liveCalibration", "extrinsicsCalibration"):
          r = extract_calibration_rpy(e, w)
          if len(r) == 3:
            cal.append((t, *r))
      seg_t.append(first)
      if verbose:
        print(f"  read {f.parent.name}/{f.name}: {len(ds)} driver-state msgs so far", file=sys.stderr)
    if not ds or not dm:
      raise RuntimeError("no driverState / driverStateV2 / driverMonitoringState messages found (is driver monitoring logged in these rlogs?)")
    ds.sort(key=lambda r: r[0]); dm.sort(key=lambda r: r[0]); cs.sort(key=lambda r: r[0]); cc.sort(key=lambda r: r[0]); cal.sort(key=lambda r: r[0])
    if any(r[1] for r in sds):     # newer openpilot: alerts live in selfdriveState
      ctl = sds
    ctl.sort(key=lambda r: r[0])
    self.t0 = min(ds[0][0], dm[0][0])
    t0 = self.t0
    self.seg_start = [s - t0 if s is not None else None for s in seg_t]
    # --- driver monitoring state
    self.tdm = np.array([r[0] - t0 for r in dm])
    self.events = [r[1] for r in dm]
    self.facedet = np.array([r[2] for r in dm], bool)
    self.isd = np.array([r[3] for r in dm], bool)
    self.dtype = np.array([r[4] for r in dm], int)
    self.aware = np.array([r[5] for r in dm], float)
    self.is_rhd = bool(np.mean([r[6] for r in dm]) > 0.5)
    pitch_off = np.array([r[7] for r in dm]); pitch_n = np.array([r[8] for r in dm])
    yaw_off = np.array([r[9] for r in dm]); yaw_n = np.array([r[10] for r in dm])
    # --- driver state (the driver's side of the car)
    side = 2 if self.is_rhd else 1
    self.tds = np.array([r[0] - t0 for r in ds])
    a = np.array([r[side] for r in ds], float)
    self.pitch_net, self.yaw_net, self.roll_net = (np.degrees(a[:, i]) for i in range(3))
    fpx, fpy = a[:, 3], a[:, 4]
    (self.faceP, self.lEye, self.rEye, self.lBl, self.rBl, self.sun, self.phone) = (a[:, i] for i in range(5, 12))
    # --- openpilot face_orientation_from_net + device calibration
    if cal:
      tc = np.array([r[0] - t0 for r in cal]); rpy = np.array([r[1:] for r in cal])
      idx = np.clip(np.searchsorted(tc, self.tds, "right") - 1, 0, len(tc) - 1)
      cal_p, cal_y = rpy[idx, 1], rpy[idx, 2]
    else:
      cal_p = cal_y = np.zeros_like(self.tds)
    self.calib_rpy_deg = np.degrees(np.median(np.array([r[1:] for r in cal]), 0)) if cal else np.zeros(3)
    yaw_focal = np.arctan2((fpx + 0.5) * CAM_W - CAM_W // 2, EFL)
    pitch_focal = np.arctan2((fpy + 0.5) * CAM_H - CAM_H // 2, EFL)
    pitch = np.radians(self.pitch_net) + pitch_focal - cal_p
    yaw = -np.radians(self.yaw_net) + yaw_focal - cal_y
    # learned offsets only once dmonitoringd has >= 1 min of samples, else natural offsets
    j = np.clip(np.searchsorted(self.tdm, self.tds, "right") - 1, 0, len(self.tdm) - 1)
    calibrated = (pitch_n[j] > POSE_OFFSET_MIN_COUNT) & (yaw_n[j] > POSE_OFFSET_MIN_COUNT)
    if uses_v111_dm(self.meta):
      nat_p, nat_y = PITCH_NATURAL_OFFSET_V111, YAW_NATURAL_OFFSET_V111
    else:
      nat_p, nat_y = PITCH_NATURAL_OFFSET, YAW_NATURAL_OFFSET
    p_off = np.where(calibrated, np.clip(pitch_off[j], PITCH_MIN_OFFSET, PITCH_MAX_OFFSET), nat_p)
    y_off = np.where(calibrated, np.clip(yaw_off[j], YAW_MIN_OFFSET, YAW_MAX_OFFSET), nat_y)
    self.pose_calibrated_frac = float(calibrated.mean())
    self.yaw_err = np.degrees(yaw - y_off)      # + = driver's right (LHD)
    self.pitch_err = np.degrees(pitch - p_off)  # - = looking down
    # --- car / alerts
    self.tcs = np.array([r[0] - t0 for r in cs]) if cs else np.array([0.0])
    self.v = np.array([r[1] for r in cs]) if cs else np.zeros(1)
    self.steer = unwrap_angle([r[2] for r in cs]) if cs else np.zeros(1)
    self.steer_pressed_raw = np.array([r[3] for r in cs], bool) if cs else np.zeros(1, bool)
    self.steer_torque = np.array([r[4] for r in cs]) if cs else np.zeros(1)
    self.steer_pressed = hold_flags(self.tcs, self.steer_pressed_raw)
    self.has_carcontrol = bool(cc)
    tcc = np.array([r[0] - t0 for r in cc]) if cc else np.zeros(0)
    self.lat_active = align_prev(tcc, np.array([r[2] for r in cc], bool), self.tcs, False).astype(bool) if cc else np.zeros(len(self.tcs), bool)
    self.cc_enabled = align_prev(tcc, np.array([r[1] for r in cc], bool), self.tcs, False).astype(bool) if cc else np.zeros(len(self.tcs), bool)
    # openpilot's commanded wheel angle (AP1 is angle control); only meaningful while lat is active
    self.steer_cmd = np.interp(self.tcs, tcc, unwrap_angle([r[3] for r in cc])) if cc else self.steer.copy()
    self.steer_who = classify_steering(self.lat_active, self.steer_pressed)
    self.tct = np.array([r[0] - t0 for r in ctl]) if ctl else np.array([0.0])
    self.alerts = [(r[1], r[2], r[3]) for r in ctl] if ctl else [("", "", "normal")]
    self.duration = float(max(self.tds[-1], self.tdm[-1]))

  # ------------------------------------------------------------------ helpers
  @staticmethod
  def _prev(ts, t):
    return max(0, int(np.searchsorted(ts, t, "right")) - 1)

  def segment_of(self, t):
    k = 0
    for i, s in enumerate(self.seg_start):
      if s is not None and s <= t + 1e-6:
        k = i
    from .logreader import _seg_index
    return _seg_index(self.files[k]), t - (self.seg_start[k] or 0.0)

  def baseline(self, start, end, pad=(15.0, 15.0)):
    """Driver's forward pose: median raw net yaw/pitch/roll in the 15 s around the window while not distracted."""
    m = (((self.tds >= start - pad[0]) & (self.tds < start)) | ((self.tds > end) & (self.tds <= end + pad[1])))
    j = np.clip(np.searchsorted(self.tdm, self.tds, "right") - 1, 0, len(self.tdm) - 1)
    m &= ~self.isd[j]
    if m.sum() < 20:
      m = ~self.isd[j]
    return float(np.median(self.yaw_net[m])), float(np.median(self.pitch_net[m])), float(np.median(self.roll_net[m]))

  def state(self, t, mph=True):
    lerp = lambda arr, ts: float(np.interp(t, ts, arr))
    i = self._prev(self.tdm, t)
    a1, a2, ast = self.alerts[self._prev(self.tct, t)]
    nxt = min(i + 1, len(self.aware) - 1)
    aw = lerp(self.aware, self.tdm) if abs(self.aware[nxt] - self.aware[i]) < 0.2 else self.aware[i]
    return dict(pn=lerp(self.pitch_net, self.tds), yn=lerp(self.yaw_net, self.tds), rn=lerp(self.roll_net, self.tds),
                ye=lerp(self.yaw_err, self.tds), pe=lerp(self.pitch_err, self.tds), face=lerp(self.faceP, self.tds),
                le=lerp(self.lEye, self.tds), re=lerp(self.rEye, self.tds), lb=lerp(self.lBl, self.tds), rb=lerp(self.rBl, self.tds),
                sun=lerp(self.sun, self.tds), ph=lerp(self.phone, self.tds), aw=float(aw), dist=bool(self.isd[i]),
                dt=int(self.dtype[i]), fdet=bool(self.facedet[i]), a1=a1, a2=a2, ast=ast,
                v=lerp(self.v, self.tcs) * (2.23694 if mph else 3.6),
                **steering_at(t, self.tcs, self.steer, self.steer_cmd, self.lat_active, self.steer_who))

  def alert_strength(self, t, kind):
    vals = []
    for dt in np.linspace(-0.18, 0, 7):
      a1, a2, _ = self.alerts[self._prev(self.tct, t + dt)]
      vals.append(1.0 if (kind == "any" and a1) or (kind == "prompt" and a2) else 0.0)
    return float(np.mean(vals))
