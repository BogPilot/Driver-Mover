"""Rank the most interesting driver-monitoring moments in a route."""
import numpy as np

EVENT_LABELS = {
  "preDriverDistracted": ("Pay Attention", 2), "promptDriverDistracted": ("Driver Distracted (orange)", 4),
  "driverDistracted": ("DISENGAGE - distracted (red)", 8), "preDriverUnresponsive": ("Touch Steering Wheel", 2),
  "promptDriverUnresponsive": ("Touch Steering Wheel (orange)", 4), "driverUnresponsive": ("DISENGAGE - unresponsive (red)", 8),
  "tooDistracted": ("Too distracted", 6),
}


def episodes(r, gap=2.0):
  active = r.isd | (r.aware < 0.999) | np.array([bool(e) for e in r.events])
  eps, s = [], None
  for i, a in enumerate(active):
    t = r.tdm[i]
    if a and s is None:
      s = t
    if not a and s is not None:
      eps.append([s, t]); s = None
  if s is not None:
    eps.append([s, r.tdm[-1]])
  merged = []
  for e in eps:
    if merged and e[0] - merged[-1][1] < gap:
      merged[-1][1] = e[1]
    else:
      merged.append(e)
  return merged


def describe(r, a, b):
  m = (r.tdm >= a) & (r.tdm <= b)
  idx = np.nonzero(m)[0]
  counts = {}
  prev = set()
  for i in idx:
    cur = set(r.events[i])
    for ev in cur - prev:
      counts[ev] = counts.get(ev, 0) + 1
    prev = cur
  ms = (r.tds >= a) & (r.tds <= b)
  return dict(counts=counts, min_aw=float(r.aware[m].min()) if m.any() else 1.0,
              max_yaw=float(np.abs(r.yaw_err[ms]).max()) if ms.any() else 0.0,
              min_pitch=float(r.pitch_err[ms].min()) if ms.any() else 0.0,
              distracted_s=float(r.isd[m].sum() * 0.05))


def score(d):
  s = sum(EVENT_LABELS.get(k, ("", 1))[1] * v for k, v in d["counts"].items())
  return s + 6 * (1 - d["min_aw"]) + d["max_yaw"] / 30 + d["distracted_s"] / 5


def windows(r, duration=15.0, top=5):
  out = []
  for a, b in episodes(r):
    L = b - a
    start = a - (duration - L) / 2 if L < duration else a - 2.0
    start = round(min(max(0.0, start), max(0.0, r.duration - duration)) * 2) / 2
    d = describe(r, start, start + duration)
    out.append(dict(start=start, end=start + duration, episode=(a, b), score=score(d), **d))
  if not out:   # no alerts at all: show the biggest head movements instead
    ye = np.interp(np.arange(0, r.duration, 0.5), r.tds, np.abs(r.yaw_err))
    used = []
    for i in np.argsort(-ye):
      t = i * 0.5
      if all(abs(t - u) > duration for u in used):
        used.append(t)
        start = round(min(max(0.0, t - duration / 2), max(0.0, r.duration - duration)) * 2) / 2
        d = describe(r, start, start + duration)
        out.append(dict(start=start, end=start + duration, episode=(t, t), score=score(d), **d))
      if len(used) >= top:
        break
  out.sort(key=lambda w: -w["score"])
  # drop overlapping windows (keep the better one)
  keep = []
  for w in out:
    if all(w["end"] <= k["start"] or w["start"] >= k["end"] for k in keep):
      keep.append(w)
  return keep[:top]


def summary(w):
  parts = [f"{n}x {EVENT_LABELS.get(k, (k,))[0]}" for k, n in sorted(w["counts"].items(), key=lambda kv: -EVENT_LABELS.get(kv[0], ("", 1))[1])]
  if w["min_aw"] < 0.95:
    parts.append(f"awareness dropped to {w['min_aw'] * 100:.0f}%")
  parts.append(f"head turned up to {w['max_yaw']:.0f}\u00b0")
  if w["min_pitch"] < -15:
    parts.append(f"head down to {-w['min_pitch']:.0f}\u00b0")
  return ", ".join(parts)
