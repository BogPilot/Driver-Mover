"""Choose a cereal schema pack so one Driver-Mover tree can read rlogs from
openpilot 0.9.7 through the current release.

Cap'n Proto ordinals are not always stable across that range, so a single
schema cannot cover every log:

* Stock 0.9.8+ puts selfdriveState at Event @130. The bundled BogPilot /
  FrogPilot schema (and some 0.9.7-era forks) puts liveDelay at @130.
* In 0.11.1 driverMonitoringState moved from @71 to @151 and the payload
  was rewritten. An older schema skips @151, so those logs look empty.
* 0.11.2 renamed liveCalibration -> extrinsicsCalibration (@19) and
  driverCameraState -> cabinCameraState (@70).

This module reads initData.version, picks a pack, and falls back to probing
if the version string is missing or the first pick cannot see DM messages.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .logreader import iter_events, load_schema

SCHEMA_DIR = Path(__file__).parent / "schema"
PACKS = {
    "bundled": SCHEMA_DIR,
    "stock_v097": SCHEMA_DIR / "packs" / "stock_v097",
    "stock_v111plus": SCHEMA_DIR / "packs" / "stock_v111plus",
}

# Prefer the newest pack first when probing: an 0.11.2 schema can still
# *name* older union members via their DEPRECATED aliases, while an older
# schema simply cannot see @151.
# 0.9.8–0.11.0 logs also use stock_v111plus: selfdriveState @130 and the
# pre-0.11.1 DriverMonitoringState at @71 (named driverMonitoringStateDEPRECATED
# in this schema) are both visible. A mid-family snapshot is not loaded —
# at v0.11.0, cereal/car.capnp is only a symlink into opendbc and pycapnp
# cannot open it.
PROBE_ORDER = ("stock_v111plus", "bundled", "stock_v097")

VERSION_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")

# initData is enough to pick a pack; DM names confirm it
_INIT_WANTED = {"initData"}
_DM_WHICH = {
    "driverStateV2", "driverState", "driverStateDEPRECATED",
    "driverMonitoringState", "driverMonitoringStateDEPRECATED",
}
_PROBE_EVENT_LIMIT = 40000


@dataclass
class LogMeta:
    version_text: str = ""
    version: tuple[int, int, int] | None = None
    git_branch: str = ""
    git_remote: str = ""
    pack: str = "bundled"
    pack_dir: Path = SCHEMA_DIR
    probed: bool = False
    custom: bool = False


def parse_op_version(text: str | None) -> tuple[int, int, int] | None:
    """Pull the first x.y.z from an initData.version string such as 'openpilot v0.11.2'."""
    if not text:
        return None
    m = VERSION_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def choose_pack_name(version: tuple[int, int, int] | None) -> str | None:
    """Map a parsed openpilot version to a pack. None means 'probe'."""
    if version is None:
        return None
    if version >= (0, 9, 8):
        return "stock_v111plus"
    if version >= (0, 9, 7):
        # The bundled BogPilot schema is a 0.9.7-era FrogPilot derivative.
        # Stock 0.9.7 DM ordinals match it (DMS @71, no selfdriveState).
        return "bundled"
    return None


def pack_dir(name: str) -> Path:
    if name not in PACKS:
        raise KeyError(f"unknown schema pack: {name}")
    return PACKS[name]



def sniff_op_version(path) -> tuple[int, int, int] | None:
    """Read initData.version without loading a cereal schema.

    Cap'n Proto text fields appear as UTF-8 in the decompressed rlog, and
    initData is the first event, so a small prefix is enough. Only one
    log.capnp can be loaded per process (they share a file id), so we must
    pick the pack *before* calling capnp.load.
    """
    from .logreader import _decompress
    try:
        data = _decompress(path)
    except Exception:
        return None
    sample = data[:2_000_000].decode("latin-1", "ignore")
    prefixed = re.search(
        r"(?:openpilot|bogpilot|sunnypilot|frogpilot)[^\x00]{0,48}?v?(\d+\.\d+\.\d+)",
        sample,
        re.I,
    )
    if prefixed:
        return parse_op_version(prefixed.group(0))
    # Fall back to a vX.Y.Z in the first 64 KiB, which is almost always initData.
    hits = VERSION_RE.findall(sample[:65536])
    if not hits:
        return None
    a, b, c = hits[0]
    return int(a), int(b), int(c)


def read_init_fields(path, schema_dir=None) -> dict:
    """Return version / git fields from the first initData event in an rlog.

    Loads `schema_dir`. Caller must not have loaded a different log.capnp
    in this process.
    """
    out = {"version_text": "", "git_branch": "", "git_remote": ""}
    try:
        for which, evt in iter_events(path, schema_dir, _INIT_WANTED, max_events=200):
            if which != "initData":
                continue
            d = evt.initData
            out["version_text"] = str(getattr(d, "version", "") or "")
            out["git_branch"] = str(getattr(d, "gitBranch", "") or "")
            out["git_remote"] = str(getattr(d, "gitRemote", "") or "")
            return out
    except Exception:
        return out
    return out


def pack_sees_driver_messages(path, schema_dir, limit=_PROBE_EVENT_LIMIT) -> bool:
    """True if this schema can name at least one driver-state or DM event.

    Loads `schema_dir`. Caller must not have loaded a different log.capnp
    in this process.
    """
    try:
        for which, _evt in iter_events(path, schema_dir, _DM_WHICH, max_events=limit):
            if which in _DM_WHICH:
                return True
    except Exception:
        return False
    return False


def select_schema(paths, schema_dir=None, verbose=True) -> LogMeta:
    """Pick a schema directory for this route. An explicit schema_dir always wins.

    At most one log.capnp is loaded in this process.
    """
    files = list(paths)
    first = files[0] if files else None
    if schema_dir is not None:
        meta = LogMeta(pack="custom", pack_dir=Path(schema_dir), custom=True)
        if first:
            fields = read_init_fields(first, schema_dir)
            meta.version_text = fields["version_text"]
            meta.version = parse_op_version(meta.version_text)
            meta.git_branch = fields["git_branch"]
            meta.git_remote = fields["git_remote"]
        if verbose:
            print(f"  schema: using --schema-dir {meta.pack_dir}", file=sys.stderr)
        return meta

    meta = LogMeta()
    sniffed = sniff_op_version(first) if first else None
    meta.version = sniffed
    if sniffed:
        meta.version_text = "v%d.%d.%d" % sniffed

    name = choose_pack_name(meta.version) or "stock_v111plus"
    probed = meta.version is None
    meta.pack = name
    meta.pack_dir = PACKS[name]
    meta.probed = probed
    if verbose:
        ver = meta.version_text or "unknown version"
        how = "sniffed" if sniffed else "defaulted"
        print(f"  schema: {how} {ver} -> pack {name}", file=sys.stderr)
    load_schema(meta.pack_dir)
    if first:
        fields = read_init_fields(first, meta.pack_dir)
        if fields["version_text"]:
            meta.version_text = fields["version_text"]
            meta.version = parse_op_version(meta.version_text) or meta.version
            meta.git_branch = fields["git_branch"]
            meta.git_remote = fields["git_remote"]
    return meta


def uses_v111_dm(meta: LogMeta) -> bool:
    """True when we should interpret DriverMonitoringState as the 0.11.1+ shape."""
    if meta.pack == "stock_v111plus" and meta.version is not None and meta.version >= (0, 11, 1):
        return True
    if meta.version is not None and meta.version >= (0, 11, 1):
        return True
    return False
