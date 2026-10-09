"""Extract driver-monitoring signals from a route and convert them like openpilot's dmonitoringd does."""
import math
import sys

import numpy as np

from .logreader import find_segments, iter_events

# openpilot selfdrive/monitoring/helpers.py constants
DT_DMON = 0.05
POSE_OFFSET_MIN_COUNT = int(60 / DT_DMON)
PITCH_NATURAL_OFFSET, YAW_NATURAL_OFFSET = 0.029, 0.097
PITCH_MIN_OFFSET, PITCH_MAX_OFFSET = -0.0881, 0.124
YAW_MIN_OFFSET, YAW_MAX_OFFSET = -0.0246, 0.289
YAW_TH = (math.degrees(0.4020), math.degrees(0.5042))   # strict, slack
PITCH_TH = math.degrees(0.449)                           # natural (uncalibrated) head-down limit
AWARE_PRE, AWARE_PROMPT = 8 / 11, 6 / 11
# face_orientation_from_net: model output refers to the undistorted 1928x1208 driver image, EFL 598 px
CAM_W, CAM_H, EFL = 1928, 1208, 598.0

WANTED = {"driverStateV2", "driverMonitoringState", "controlsState", "selfdriveState", "carState", "liveCalibration"}


def _g(obj, name, default=0.0):
  try:
    return getattr(obj, name)
  except Exception:
    return default


class Route:
  def __init__(self, paths, schema_dir=None, verbose=True):
    self.files = find_segments(paths)
    ds, dm, cs, ctl, sds, cal, seg_t = [], [], [], [], [], [], []
    for f in self.files:
      first = None
      for w, e in iter_events(f, schema_dir, WANTED):
        t = e.logMonoTime * 1e-9
        if w == "driverStateV2":
          d = e.driverStateV2
          def dd(x):
            return (list(x.faceOrientation) + list(x.facePosition)[:2] + [x.faceProb, x.leftEyeProb, x.rightEyeProb,
                    x.leftBlinkProb, x.rightBlinkProb, x.sunglassesProb, _g(x, "phoneProb")])
          ds.append((t, dd(d.leftDriverData), dd(d.rightDriverData), d.wheelOnRightProb))
          first = t if first is None else min(first, t)
        elif w == "driverMonitoringState":
          m = e.driverMonitoringState
          dm.append((t, [str(x.name) for x in m.events], m.faceDetected, m.isDistracted, m.distractedType,
                     m.awarenessStatus, m.isRHD, m.posePitchOffset, m.posePitchValidCount, m.poseYawOffset, m.poseYawValidCount))
        elif w in ("controlsState", "selfdriveState"):
          c = getattr(e, w)
          row = (t, str(_g(c, "alertText1", "")), str(_g(c, "alertText2", "")), str(_g(c, "alertStatus", "normal")))
          (ctl if w == "controlsState" else sds).append(row)
        elif w == "carState":
          c = e.carState
          cs.append((t, c.vEgo, c.steeringAngleDeg))
        elif w == "liveCalibration":
          r = list(e.liveCalibration.rpyCalib)
          if len(r) == 3:
            cal.append((t, *r))
      seg_t.append(first)
      if verbose:
        print(f"  read {f.parent.name}/{f.name}: {len(ds)} driver-state msgs so far", file=sys.stderr)
    if not ds or not dm:
      raise RuntimeError("no driverStateV2 / driverMonitoringState messages found (is driver monitoring logged in these rlogs?)")
    ds.sort(key=lambda r: r[0]); dm.sort(key=lambda r: r[0]); cs.sort(key=lambda r: r[0]); cal.sort(key=lambda r: r[0])
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
    # learned offsets only once dmonitoringd has >= 1 min of samples, else natural offsets (as in helpers.py)
    j = np.clip(np.searchsorted(self.tdm, self.tds, "right") - 1, 0, len(self.tdm) - 1)
    calibrated = (pitch_n[j] > POSE_OFFSET_MIN_COUNT) & (yaw_n[j] > POSE_OFFSET_MIN_COUNT)
    p_off = np.where(calibrated, np.clip(pitch_off[j], PITCH_MIN_OFFSET, PITCH_MAX_OFFSET), PITCH_NATURAL_OFFSET)
    y_off = np.where(calibrated, np.clip(yaw_off[j], YAW_MIN_OFFSET, YAW_MAX_OFFSET), YAW_NATURAL_OFFSET)
    self.pose_calibrated_frac = float(calibrated.mean())
    self.yaw_err = np.degrees(yaw - y_off)      # + = driver's right (LHD)
    self.pitch_err = np.degrees(pitch - p_off)  # - = looking down
    # --- car / alerts
    self.tcs = np.array([r[0] - t0 for r in cs]) if cs else np.array([0.0])
    self.v = np.array([r[1] for r in cs]) if cs else np.zeros(1)
    self.steer = np.array([r[2] for r in cs]) if cs else np.zeros(1)
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
                v=lerp(self.v, self.tcs) * (2.23694 if mph else 3.6), st=lerp(self.steer, self.tcs))

  def alert_strength(self, t, kind):
    vals = []
    for dt in np.linspace(-0.18, 0, 7):
      a1, a2, _ = self.alerts[self._prev(self.tct, t + dt)]
      vals.append(1.0 if (kind == "any" and a1) or (kind == "prompt" and a2) else 0.0)
    return float(np.mean(vals))
