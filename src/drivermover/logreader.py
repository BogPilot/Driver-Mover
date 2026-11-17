"""Standalone rlog reader: pycapnp + a bundled cereal schema (no openpilot checkout needed)."""
import bz2
import io
import os
import re
from pathlib import Path

import capnp

SCHEMA_DIR = Path(__file__).parent / "schema"
_LOG = {}


def load_schema(schema_dir=None):
  """Load cereal log.capnp (bundled BogPilot copy by default, or a pack / fork cereal folder)."""
  d = str(Path(schema_dir) if schema_dir else SCHEMA_DIR)
  if d not in _LOG:
    capnp.remove_import_hook()
    _LOG[d] = capnp.load(os.path.join(d, "log.capnp"), imports=[d])
  return _LOG[d]


def _decompress(path):
  data = Path(path).read_bytes()
  if data[:4] == b"\x28\xb5\x2f\xfd":  # zstd
    import zstandard
    return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()
  if data[:3] == b"BZh":
    return bz2.decompress(data)
  return data


# 0.9.7: rlog, rlog.bz2, raw_log.bz2 (uncompressed rlog also exists)
# 0.9.8+: rlog.zst / qlog.zst. Official 0.11 FileName lists rlog.zst and rlog.bz2.
# Connect downloads often use <route>--<seg>--rlog.zst as a flat name.
_RLOG = re.compile(r"^(rlog|raw_log)(\.bz2|\.zst)?$")
_RLOG_SUFFIX = re.compile(r"--(rlog|raw_log)(\.bz2|\.zst)?$")


def _seg_index(p):
  # .../<route>--<seg>/rlog  or  .../<route>--<seg>--rlog.zst  or a plain file
  for part in (p.parent.name, p.name):
    m = re.search(r"--(\d+)(?:--(?:rlog|raw_log).*)?$", part)
    if m:
      return int(m.group(1))
  # 0.11-style: .../<route>/<seg>/rlog.zst where <seg> is a bare integer folder
  if re.fullmatch(r"\d+", p.parent.name or ""):
    return int(p.parent.name)
  return 0


def find_segments(paths):
  """Accept a route folder, segment folders, or rlog files. Returns rlog files sorted by segment."""
  files = []
  for p in map(Path, paths if isinstance(paths, (list, tuple)) else [paths]):
    p = p.expanduser()
    if p.is_file():
      files.append(p)
    elif p.is_dir():
      for f in p.rglob("*"):
        if f.is_file() and (_RLOG.match(f.name) or _RLOG_SUFFIX.search(f.name)):
          files.append(f)
    else:
      raise FileNotFoundError(f"not found: {p}")
  if not files:
    raise FileNotFoundError("no rlog / rlog.zst / rlog.bz2 / raw_log.bz2 files found in " + ", ".join(map(str, paths if isinstance(paths, (list, tuple)) else [paths])))
  return sorted(set(files), key=lambda f: (_seg_index(f), str(f)))


def iter_events(path, schema_dir=None, wanted=None, max_events=None):
  """Yield (union_name, event) for events this schema can name.

  Unknown Event-union members (added in a newer openpilot than the schema)
  are skipped. max_events, if set, counts every Event including skipped ones.
  """
  log = load_schema(schema_dir)
  data = _decompress(path)
  n = 0
  for evt in log.Event.read_multiple_bytes(data, traversal_limit_in_words=2**62):
    n += 1
    if max_events is not None and n > max_events:
      return
    try:
      w = evt.which()
    except Exception:  # union member unknown to this schema
      continue
    if wanted is None or w in wanted:
      yield w, evt
