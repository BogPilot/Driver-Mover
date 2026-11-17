"""Schema-pack detection and cross-version message normalization (no real rlogs).

Each case that calls capnp.load on a log.capnp runs in its own subprocess:
all of these schemas share the same file id, so only one can be loaded per
process.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from drivermover.logreader import _seg_index, find_segments
from drivermover.schema_packs import PACKS, choose_pack_name, parse_op_version


def test_parse_and_choose_pack():
    assert parse_op_version("openpilot v0.9.7") == (0, 9, 7)
    assert parse_op_version("0.10.3") == (0, 10, 3)
    assert parse_op_version("bogpilot-tesla based on 0.11.2") == (0, 11, 2)
    assert parse_op_version("") is None
    assert choose_pack_name((0, 9, 7)) == "bundled"
    assert choose_pack_name((0, 9, 8)) == "stock_v111plus"
    assert choose_pack_name((0, 9, 9)) == "stock_v111plus"
    assert choose_pack_name((0, 10, 0)) == "stock_v111plus"
    assert choose_pack_name((0, 10, 3)) == "stock_v111plus"
    assert choose_pack_name((0, 11, 0)) == "stock_v111plus"
    assert choose_pack_name((0, 11, 1)) == "stock_v111plus"
    assert choose_pack_name((0, 11, 2)) == "stock_v111plus"
    assert choose_pack_name((0, 12, 0)) == "stock_v111plus"
    assert choose_pack_name(None) is None


def _isolated(code: str) -> None:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(r.stdout + "\n" + r.stderr)


@pytest.mark.parametrize("name", list(PACKS))
def test_pack_loads(name):
    _isolated(f"""
from drivermover.logreader import load_schema
from drivermover.schema_packs import PACKS
log = load_schema(PACKS[{name!r}])
assert hasattr(log, "Event"), {name!r}
print("OK", {name!r})
""")


def test_bundled_roundtrip_v2_and_old_dms():
    _isolated(r"""
from pathlib import Path
import tempfile
from drivermover.logreader import iter_events
from drivermover.schema_packs import PACKS, select_schema

schema = PACKS["bundled"]

def write(path):
    from drivermover.logreader import load_schema
    log = load_schema(schema)
    chunks = []
    def add(fn):
        evt = log.Event.new_message()
        evt.logMonoTime = 10**9
        fn(evt)
        chunks.append(evt.to_bytes())
    def init(evt):
        evt.init("initData").version = "openpilot v0.9.7"
    def v2(evt):
        d = evt.init('driverStateV2')
        d.leftDriverData.faceOrientation = [0.1, 0.2, 0.0]
        d.leftDriverData.facePosition = [0.0, 0.0]
        d.leftDriverData.faceProb = 0.9
        d.rightDriverData.faceOrientation = [0.0, 0.0, 0.0]
        d.rightDriverData.facePosition = [0.0, 0.0]
        d.rightDriverData.faceProb = 0.1
        d.wheelOnRightProb = 0.05
    def dms(evt):
        m = evt.init('driverMonitoringState')
        m.faceDetected = True
        m.isDistracted = False
        m.awarenessStatus = 0.8
        m.isRHD = False
        m.posePitchOffset = 0.02
        m.posePitchValidCount = 2000
        m.poseYawOffset = 0.1
        m.poseYawValidCount = 2000
    for fn in (init, v2, dms):
        add(fn)
    path.write_bytes(b"".join(chunks))

with tempfile.TemporaryDirectory() as td:
    rlog = Path(td) / "rlog"
    write(rlog)
    whichs = [w for w, _ in iter_events(rlog, schema)]
    assert "initData" in whichs
    assert "driverStateV2" in whichs
    assert "driverMonitoringState" in whichs
    meta = select_schema([rlog], verbose=False)
    assert meta.pack == "bundled", meta
    assert meta.version == (0, 9, 7), meta
print("OK bundled")
""")


def test_v110_selfdrive_state_and_phone_prob():
    _isolated(r"""
import math, tempfile
from pathlib import Path
import zstandard
from drivermover.data import extract_driver_state
from drivermover.logreader import iter_events, load_schema
from drivermover.schema_packs import PACKS, select_schema

schema = PACKS["stock_v111plus"]

def write(path):
    log = load_schema(schema)
    chunks = []
    def add(fn):
        evt = log.Event.new_message()
        evt.logMonoTime = 10**9
        fn(evt)
        chunks.append(evt.to_bytes())
    def init(evt):
        evt.init("initData").version = "openpilot v0.10.2"
    def v2(evt):
        d = evt.init('driverStateV2')
        d.leftDriverData.faceOrientation = [0.1, 0.2, 0.0]
        d.leftDriverData.facePosition = [0.0, 0.0]
        d.leftDriverData.faceProb = 0.95
        d.leftDriverData.phoneProb = 0.4
        d.rightDriverData.faceOrientation = [0.0, 0.0, 0.0]
        d.rightDriverData.facePosition = [0.0, 0.0]
    def dms(evt):
        m = evt.init('driverMonitoringStateDEPRECATED')
        m.faceDetected = True
        m.isDistracted = True
        m.distractedType = 4
        m.awarenessStatus = 0.5
        m.isRHD = False
    def sds(evt):
        s = evt.init('selfdriveState')
        s.alertText1 = "PAY ATTENTION"
        s.alertText2 = "Driver Distracted"
    for fn in (init, v2, dms, sds):
        add(fn)
    raw = b"".join(chunks)
    path.write_bytes(zstandard.ZstdCompressor().compress(raw))

with tempfile.TemporaryDirectory() as td:
    rlog = Path(td) / "rlog.zst"
    write(rlog)
    evts = list(iter_events(rlog, schema))
    whichs = [w for w, _ in evts]
    assert "selfdriveState" in whichs
    assert "driverStateV2" in whichs
    v2_evt = next(e for w, e in evts if w == "driverStateV2")
    left, right, _wor = extract_driver_state("driverStateV2", v2_evt)
    assert math.isclose(left[11], 0.4, rel_tol=1e-6)
    meta = select_schema([rlog], verbose=False)
    assert meta.pack == "stock_v111plus", meta
    assert meta.version == (0, 10, 2), meta
print("OK v110")
""")


def test_v111plus_new_dms():
    _isolated(r"""
import math, tempfile
from pathlib import Path
from drivermover.data import DISTRACT_PHONE, DISTRACT_POSE, extract_dms
from drivermover.logreader import iter_events, load_schema
from drivermover.schema_packs import PACKS, select_schema

schema = PACKS["stock_v111plus"]

def write(path):
    log = load_schema(schema)
    chunks = []
    def add(fn):
        evt = log.Event.new_message()
        evt.logMonoTime = 10**9
        fn(evt)
        chunks.append(evt.to_bytes())
    def init(evt):
        evt.init("initData").version = "openpilot v0.11.2"
    def v2(evt):
        d = evt.init('driverStateV2')
        d.leftDriverData.faceOrientation = [0.05, -0.1, 0.0]
        d.leftDriverData.facePosition = [0.01, -0.02]
        d.leftDriverData.faceProb = 0.99
        d.leftDriverData.phoneProb = 0.2
        d.leftDriverData.sleepProb = 0.01
        d.rightDriverData.faceOrientation = [0.0, 0.0, 0.0]
        d.rightDriverData.facePosition = [0.0, 0.0]
        d.wheelOnRightProb = 0.1
    def dms(evt):
        m = evt.init('driverMonitoringState')
        m.isRHD = False
        m.alertLevel = "two"
        m.activePolicy = "vision"
        v = m.visionPolicyState
        v.awarenessPercent = 61
        v.isDistracted = True
        v.faceDetected = True
        v.distractedTypes.pose = True
        v.distractedTypes.phone = True
        v.pose.calibrated = True
        v.pose.pitchCalib.offset = 0.02
        v.pose.pitchCalib.calibratedPercent = 100
        v.pose.yawCalib.offset = 0.08
        v.pose.yawCalib.calibratedPercent = 100
    def cal(evt):
        c = evt.init('extrinsicsCalibration')
        c.rpyCalib = [0.0, 0.01, -0.02]
    for fn in (init, v2, dms, cal):
        add(fn)
    path.write_bytes(b"".join(chunks))

with tempfile.TemporaryDirectory() as td:
    rlog = Path(td) / "rlog"
    write(rlog)
    evts = list(iter_events(rlog, schema))
    whichs = [w for w, _ in evts]
    assert "driverMonitoringState" in whichs
    assert "extrinsicsCalibration" in whichs
    assert "liveCalibration" not in whichs
    dms_evt = next(e for w, e in evts if w == "driverMonitoringState")
    row = extract_dms("driverMonitoringState", dms_evt)
    events, face, isd, dtype, aw, is_rhd, p_off, p_n, y_off, y_n = row
    assert face is True
    assert isd is True
    assert dtype & DISTRACT_POSE
    assert dtype & DISTRACT_PHONE
    assert math.isclose(aw, 0.61, rel_tol=1e-6)
    assert is_rhd is False
    assert math.isclose(p_off, 0.02, rel_tol=1e-5, abs_tol=1e-6)
    assert p_n >= 1200
    meta = select_schema([rlog], verbose=False)
    assert meta.pack == "stock_v111plus", meta
    assert meta.version == (0, 11, 2), meta
print("OK v111")
""")


def test_find_segments_layouts(tmp_path):
    d0 = tmp_path / "testdrive--0"
    d0.mkdir()
    (d0 / "rlog").write_bytes(b"x")
    d1 = tmp_path / "testdrive--1"
    d1.mkdir()
    (d1 / "rlog.zst").write_bytes(b"x")
    d2 = tmp_path / "testdrive--2"
    d2.mkdir()
    (d2 / "raw_log.bz2").write_bytes(b"x")
    flat = tmp_path / "testdrive--5--rlog.zst"
    flat.write_bytes(b"x")
    numeric = tmp_path / "routebox" / "3"
    numeric.mkdir(parents=True)
    (numeric / "rlog.bz2").write_bytes(b"x")

    files = find_segments(tmp_path)
    idxs = [_seg_index(f) for f in files]
    assert idxs == sorted(idxs)
    names = {f.name for f in files}
    assert "rlog" in names
    assert "rlog.zst" in names
    assert "raw_log.bz2" in names
    assert "testdrive--5--rlog.zst" in names
    assert "rlog.bz2" in names
    assert 0 in idxs and 1 in idxs and 2 in idxs and 3 in idxs and 5 in idxs
