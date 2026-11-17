# Draft: multi-version rlog support

This is a **draft** of the documentation update for Driver-Mover. It has not
replaced README.md yet. It describes work in progress that teaches Driver-Mover to read
driver-monitoring rlogs from every openpilot release from **0.9.7 through the
current release (0.11.2 as of 17 Nov 2026, America/New_York)**, including
0.9.8, 0.9.9, all 0.10.x and all 0.11.x, and to keep working on newer logs
whose driver-monitoring structs still match 0.11.2 (as they do on
commaai/openpilot master as of that date).

Driver-Mover was originally designed for openpilot 0.9.7 rlogs and bundled a
single BogPilot / FrogPilot cereal snapshot from that era.

## Version support

| openpilot version | Schema pack | How it was checked | Notes |
|---|---|---|---|
| 0.9.7 | `bundled` (original schema/, unchanged) | real rlogs on a test machine (85 groups sniffed; 8 parsed; 1 full Route) + unit test | `driverStateV2` @92, `driverMonitoringState` @71. `driverState` (now `driverStateDEPRECATED` @59) is accepted as a fallback. File names: `rlog`, `rlog.bz2`, `raw_log.bz2`. |
| 0.9.8, 0.9.9 | `stock_v111plus` | schema + unit test only (no rlogs of these versions on the machine) | `selfdriveState` appears at Event @130 (alerts move here). `rlog.zst` is introduced. DM `events` move to field @18. Read with the 0.11.2 pack, which names the still-at-@71 message `driverMonitoringStateDEPRECATED`. |
| 0.10.0, 0.10.1, 0.10.2, 0.10.3 | `stock_v111plus` | schema + unit test only (no rlogs of these versions on the machine) | 0.10.2 adds `DriverData.phoneProb` @13. 0.10.2 also adds phone-prob offsets on `driverMonitoringState` (deprecated again in 0.11.0). Same pack as 0.11.2; older field names are DEPRECATED aliases. |
| 0.11.0 | `stock_v111plus` | schema + unit test only (no rlogs of this version on the machine) | Last release with the original `DriverMonitoringState` payload at @71. |
| 0.11.1 | `stock_v111plus` (cereal from tag v0.11.2) | schema + unit test only (no rlogs of this version on the machine) | **Breaking:** `driverMonitoringState` moves to @151 and is rewritten (vision / wheeltouch policy, `awarenessPercent`, nested pose calibration). The old message is `driverMonitoringStateDEPRECATED` @71. `sleepProb` is added on `DriverData`. A pre-0.11.1 schema will skip @151 and appear to find no DM messages. |
| 0.11.2 (current release) | `stock_v111plus` | real rlogs on a test machine (129 groups sniffed; 8 parsed; 1 full Route) + unit test | `liveCalibration` renamed `extrinsicsCalibration` (@19). `driverCameraState` renamed `cabinCameraState` (@70). New lockout fields and a `sleep` distracted type. Cabin-undistorted model frame is still 1928×1208, EFL 598 px. Natural pose offsets change to 0.011 / 0.075 rad. |
| Newer than 0.11.2 | `stock_v111plus`, then probe | schema only (master as of 17 Nov 2026 matches 0.11.2 DM structs) | If `initData.version` is newer, Driver-Mover still starts with this pack and probes the others if DM messages are not visible. A later cereal break will need a new pack. |

“Schema + unit test” means the pack was loaded, synthetic events were built
from that generation’s structs, and detection / normalization were asserted.
The “real rlogs” column is filled from whatever rlogs were already on the
machine, without copying them into this repo.

## How detection works

1. Read `initData.version` from the first rlog (the `initData` Event member at
   @1 is stable for this whole range).
2. Map the `x.y.z` version to a pack (table above).
3. If the version string is missing, or the chosen pack cannot name any
   `driverState*` / `driverMonitoringState*` events, probe the packs from
   newest to oldest and keep the first that can.
4. `--schema-dir` still wins, and still points at a fork’s own `cereal/`
   folder when you need it.

## What changed in the code (summary)

* `schema/packs/` — new snapshots of stock cereal (v0.9.7 and v0.11.2).
  The original `schema/` files are not replaced. 0.9.8–0.11.0 logs use the
  v0.11.2 pack (backward-compatible aliases). A v0.11.0 snapshot is kept on
  disk but not loaded: its `car.capnp` is only an opendbc symlink stub.
* `schema_packs.py` — version parsing, pack selection, probe fallback.
* `logreader.py` — accepts `raw_log.bz2`, numeric segment folders
  (`…/<seg>/rlog.zst`), and the existing `--N` and `--rlog.zst` layouts;
  `iter_events` can stop after `max_events`.
* `data.py` — branches on the Event name and the payload shape:
  * `driverStateV2` (preferred) or `driverState` / `driverStateDEPRECATED`
  * old `driverMonitoringState` / `driverMonitoringStateDEPRECATED` vs the
    0.11.1+ nested policy struct
  * alerts from `selfdriveState` when present, else `controlsState`
  * calibration from `liveCalibration` or `extrinsicsCalibration`
* Natural pose offsets follow 0.11.1+ when that generation is detected.
* Location / GPS fields are still never read.

## Things this draft does not change

* README.md itself (this file is the draft).
* The mannequin / overlay / hands behaviour, except that it now receives
  normalized DM values from newer logs.
* Panda, devices, or anything outside this repo.

