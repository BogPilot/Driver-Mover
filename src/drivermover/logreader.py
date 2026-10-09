"""Standalone rlog reader: pycapnp + the bundled cereal schema (no openpilot checkout needed)."""
import bz2
import io
import os
import re
from pathlib import Path

import capnp

SCHEMA_DIR = Path(__file__).parent / "schema"
_LOG = {}


def load_schema(schema_dir=None):
  """Load cereal log.capnp (bundled BogPilot copy by default, or a fork's cereal folder)."""
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


_RLOG = re.compile(r"^rlog(\.bz2|\.zst)?$")


def _seg_index(p):
  # .../<route>--<seg>/rlog  or  .../<route>--<seg>--rlog.zst  or a plain file
  for part in (p.parent.name, p.name):
    m = re.search(r"--(\d+)(?:--rlog.*)?$", part)
    if m:
      return int(m.group(1))
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
        if f.is_file() and (_RLOG.match(f.name) or re.search(r"--rlog(\.bz2|\.zst)?$", f.name)):
          files.append(f)
    else:
      raise FileNotFoundError(f"not found: {p}")
  if not files:
    raise FileNotFoundError("no rlog / rlog.zst / rlog.bz2 files found in " + ", ".join(map(str, paths if isinstance(paths, (list, tuple)) else [paths])))
  return sorted(set(files), key=lambda f: (_seg_index(f), str(f)))


def iter_events(path, schema_dir=None, wanted=None):
  log = load_schema(schema_dir)
  data = _decompress(path)
  for evt in log.Event.read_multiple_bytes(data, traversal_limit_in_words=2**62):
    try:
      w = evt.which()
    except Exception:  # union member unknown to this schema
      continue
    if wanted is None or w in wanted:
      yield w, evt
