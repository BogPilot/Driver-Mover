# Multi-version rlog support

Driver-Mover was originally written for openpilot **0.9.7** rlogs and bundled a
single BogPilot / FrogPilot cereal snapshot from that era. It now reads
driver-monitoring rlogs from **every openpilot release from 0.9.7 through the
current release (0.11.2 as of 17 November 2026)**: 0.9.8, 0.9.9, all 0.10.x
and all 0.11.x. Newer logs whose driver-monitoring structs still match 0.11.2
(as they do on commaai/openpilot master as of that date) use the same pack.

The original bundled schema files are unchanged. Driver-Mover sniffs
`initData.version` from the rlog, then loads one matching cereal pack. Only
one `log.capnp` can be loaded in a process (they share a capnp file id), so
the pack is chosen before any schema is loaded. `--schema-dir` still overrides
auto-detection if you need a fork's own `cereal/` folder.

Location and GPS fields in the logs are never read.

## Version support

| openpilot version | Schema pack | How it was checked | What changed for driver monitoring |
|---|---|---|---|
| 0.9.7 | `bundled` (the original `schema/`, unchanged) | Real rlogs (85 route groups sniffed, 8 parsed, one full ~60 s `Route()` load) plus unit tests | `driverStateV2` @92, `driverMonitoringState` @71. The older `driverState` message (now `driverStateDEPRECATED` @59) is accepted as a fallback. File names: `rlog`, `rlog.bz2`, `raw_log.bz2`. |
| 0.9.8, 0.9.9 | `stock_v111plus` | Schema and unit tests only (no rlogs of these versions were available) | `selfdriveState` appears at Event @130 (alert text moves here). `rlog.zst` is introduced. DM `events` move to field @18. The 0.11.2 pack still sees the message at @71, under the name `driverMonitoringStateDEPRECATED`. |
| 0.10.0, 0.10.1, 0.10.2, 0.10.3 | `stock_v111plus` | Schema and unit tests only (no rlogs of these versions were available) | 0.10.2 adds `DriverData.phoneProb` @13, and short-lived phone-prob offsets on `driverMonitoringState` (deprecated again in 0.11.0). Older field names are DEPRECATED aliases in this pack. |
| 0.11.0 | `stock_v111plus` | Schema and unit tests only (no rlogs of this version were available) | Last release with the original `DriverMonitoringState` payload at @71. |
| 0.11.1 | `stock_v111plus` (cereal from tag v0.11.2) | Schema and unit tests only (no rlogs of this version were available) | **Breaking:** `driverMonitoringState` moves to @151 and is rewritten (vision / wheeltouch policy, `awarenessPercent`, nested pose calibration). The old message is `driverMonitoringStateDEPRECATED` @71. `sleepProb` is added on `DriverData`. A pre-0.11.1 schema skips @151 and appears to find no DM messages. Driver-Mover normalizes the new tree back to the same awareness / distraction / pose-offset fields. |
| 0.11.2 (current release) | `stock_v111plus` | Real rlogs (129 route groups sniffed, 8 parsed, one full ~59 s `Route()` load) plus unit tests | `liveCalibration` is renamed `extrinsicsCalibration` (@19). `driverCameraState` is renamed `cabinCameraState` (@70). New lockout fields and a `sleep` distracted type. The cabin-undistorted model frame is still 1928×1208, EFL 598 px. Natural pose offsets change to 0.011 / 0.075 rad. |
| Newer than 0.11.2 | `stock_v111plus` | Schema only (commaai/openpilot master as of 17 November 2026 matches the 0.11.2 DM structs) | If `initData.version` is newer, this pack is still used. A later cereal break will need a new pack. |

“Schema and unit tests only” means the pack was loaded, synthetic events were
built from that generation’s structs, and detection / normalization were
asserted. Real rlogs were already on the test machine; they are not part of
this repository.

A v0.11.0 cereal snapshot is kept under `schema/packs/stock_v098_v110/` but
is not loaded: at that tag, `cereal/car.capnp` is only a symlink stub into
opendbc, which pycapnp cannot compile. 0.9.8–0.11.0 logs are read with the
0.11.2 pack instead.

## File and folder layouts

These are all accepted (a route folder, segment folders, or individual files):

* `rlog` (uncompressed, 0.9.7)
* `rlog.bz2` / `raw_log.bz2` (0.9.7-era and uploaded copies)
* `rlog.zst` (0.9.8 and later)
* Segment folders named `<route>--<N>/` (date-style or hash-style)
* Connect-style flat names: `<route>--<N>--rlog.zst`
* Numeric segment folders: `<route>/<N>/rlog.zst`

qlogs are too sparse and are not used.

## How detection works

1. The first rlog is decompressed and `initData.version` is sniffed from the
   bytes (initData is the first event; no schema is loaded yet).
2. The `x.y.z` version is mapped to a pack (table above). 0.9.7 uses
   `bundled`; 0.9.8 and later use `stock_v111plus`. If the version string is
   missing, `stock_v111plus` is the default.
3. That one pack is loaded and the route is read. Event names are accepted
   under both current and DEPRECATED aliases (`driverMonitoringState` /
   `driverMonitoringStateDEPRECATED`, `liveCalibration` /
   `extrinsicsCalibration`, `driverState` / `driverStateV2`, `controlsState`
   / `selfdriveState`).
4. `--schema-dir /path/to/fork/cereal` skips auto-detection.

## Code map

* `schema/packs/stock_v097/` — stock cereal from openpilot tag v0.9.7
  (available; 0.9.7 logs normally use the bundled pack, which matches these
  DM ordinals).
* `schema/packs/stock_v111plus/` — stock cereal from openpilot tag v0.11.2,
  plus `car.capnp` from the opendbc pin recorded on that tag.
* `schema_packs.py` — version sniffing and pack selection.
* `logreader.py` — segment discovery and decompression.
* `data.py` — normalizes each generation of driver-state and
  driver-monitoring messages, including the 0.11.1 rewrite of
  `DriverMonitoringState` @151.

The mannequin / overlay / hands behaviour is unchanged, other than receiving
normalized values from newer logs.
