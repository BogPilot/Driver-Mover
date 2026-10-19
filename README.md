# Driver-Mover

**Turn the driver-monitoring data from your openpilot / BogPilot drives into a video** — a heads-up display and a
3-D head model drawn on a photo, or a photo whose head actually turns the way the driver's did.

<p align="center">
  <a href="docs/example_mirror_checks.mp4"><img src="docs/example_mirror_checks.gif" width="360" alt="Driver-Mover example: driver glances left, right and left again before a turn"></a><br>
  <sub><b>Example</b> — 15 s at an intersection: a glance left (~−26°), a big look right (~+54°), left again, then the right turn.
  No alerts; awareness stays at 100 %. Rendered with <code>--realhead</code> on the bundled mannequin photo.
  <a href="docs/example_mirror_checks.mp4">Full-quality MP4</a>.</sub>
</p>

Driver-Mover reads the **rlogs** your comma device records and shows what the driver-monitoring (DM) system
saw: head yaw/pitch/roll, face/eye/blink/phone scores, the awareness meter, the distraction zone and any
“Pay Attention” / “Driver Distracted” alerts.

* **Overlay mode** (default) – HUD + 3-D wireframe head on any photo.
* **Realistic-head mode** (`--realhead`) – cross-dissolves between “keyframe” images of the same person in
  different head poses, so the person in the photo moves like the driver did. The head swap only touches the
  head/neck area; seatbelt and body stay pixel-for-pixel the original photo.
* **Steering wheel** (v2, on by default) – the wheel turns with the logged steering angle (openpilot's and
  yours) and, on the bundled mannequin, **the hands grip the rim and turn it**, shuffling push-pull style on
  big turns (`--hands on`, the default). A tick on the rim marks the wheel's 12 o'clock, a small panel says who
  is steering (openpilot / driver / driver override) and a ghost tick shows the angle openpilot commanded.
  `--hands off` keeps the photo still and only shows the angle on the rim; `--no-wheel` turns it all off.
* `find` – lists the most interesting moments in a drive automatically.

Everything runs **locally on your computer, on the CPU**. No full openpilot checkout is needed (the log
schemas are bundled). Location/GPS data in the logs is never read.

> Driver-Mover is an independent hobby project. It is **not affiliated with or endorsed by comma.ai**.
> It is a visualisation toy, not a safety tool.

---

> **New in v2:** the steering angle recorded in your rlogs (openpilot's steering and yours) turns the wheel:
> the mannequin's hands steer it push-pull style, a tick on the rim turns with it and a panel shows
> who is steering (see §4).

## Contents

1. [Install](#1-install)
2. [Get your rlogs](#2-get-your-rlogs)
3. [Find interesting moments](#3-find-interesting-moments)
4. [Render a video](#4-render-a-video)
5. [Use your own photo](#5-use-your-own-photo)
6. [Make keyframes for realistic-head mode](#6-make-keyframes-for-realistic-head-mode)
7. [Settings file reference](#7-settings-file-reference)
8. [How the numbers are computed (assumptions)](#8-how-the-numbers-are-computed-assumptions)
9. [Troubleshooting](#9-troubleshooting)
10. [Project layout](#10-project-layout)
11. [Licenses and credits](#11-licenses-and-credits)

---

## 1. Install

All commands in this section run **in a terminal on your own computer** (not on the comma device).

### System packages

**Linux (Ubuntu / Debian):**

```bash
sudo apt update
sudo apt install git ffmpeg libgles2 libegl1 python3-venv
```

(`libgles2`/`libegl1` are graphics libraries the MediaPipe face-landmark tool needs, even on the CPU.)

**macOS** (with [Homebrew](https://brew.sh)):

```bash
brew install git ffmpeg python@3.12
```

### Driver-Mover itself

```bash
git clone https://github.com/BogPilot/Driver-Mover.git
cd Driver-Mover
python3 -m venv .venv            # macOS with Homebrew: python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
drivermover --help
```

This installs numpy, OpenCV, Pillow, pycapnp, zstandard and MediaPipe into the private `.venv` folder
(≈150 MB, no GPU needed). Python 3.10–3.13 are supported (tested on Linux with 3.12 and 3.13; macOS uses
the same pip packages).

**Each time you open a new terminal**, re-activate it first:

```bash
cd Driver-Mover
source .venv/bin/activate
```

The command is `drivermover`; `dmoverlay` is an alias for the same thing.

---

## 2. Get your rlogs

A drive (“route”) is split into ~1-minute **segments**. Each segment folder holds an `rlog` file (newer
versions: `rlog.zst`; uploaded copies: `rlog.bz2` or `rlog.zst`). Driver-Mover needs **rlogs**
(full logs) — `qlog`s are too sparse. Segment folders look like

```
0000abcd--0123456789--0
0000abcd--0123456789--1
...
```

(older software uses date-style names such as `2024-05-01--12-34-56--0`; both work). Put all segments of one
route into one folder on your computer, e.g. `~/rlogs/my_drive/`.

### (a) Copy straight from the comma device over SSH (fastest, no upload needed)

One-time, **on the comma device's screen**: *Settings → Developer* → enable **SSH**, and under *SSH Keys*
enter your GitHub username (the device then trusts the SSH keys on your GitHub account). Find the device's
IP address under *Settings → Network → Advanced* (Wi-Fi) — below it is written as `<device-ip>`.
Your computer and the device must be on the same network.

The device only accepts **key-based logins**: there is no password, so SSH must use the private key that
matches a public key on your GitHub account. Tell `ssh`/`scp` which key to use with `-i <path-to-private-key>`.
On most computers that's `~/.ssh/id_ed25519` (or `~/.ssh/id_rsa`); the commands below store it in a `KEY`
variable so you only set it once. If you don't have a key yet, create one with `ssh-keygen -t ed25519`,
add the `.pub` file to GitHub (*Settings → SSH and GPG keys*), then re-enter your GitHub username on the device.

Run these **on your own computer** (the device is the *source*, your computer is the *destination*):

```bash
# path to the private key whose public half is on your GitHub account
KEY=~/.ssh/id_ed25519

# list the routes stored on the device (newest last)
ssh -i "$KEY" comma@<device-ip> 'ls /data/media/0/realdata/'

# copy every segment of one route into ~/rlogs/my_drive/
mkdir -p ~/rlogs/my_drive
scp -i "$KEY" -r 'comma@<device-ip>:/data/media/0/realdata/0000abcd--0123456789--*' ~/rlogs/my_drive/
```

Replace `0000abcd--0123456789` with the route name from the list. The device deletes old routes when its
storage fills up, so copy interesting drives soon. If you only want the logs (not the large camera videos),
copy just the rlog files:

```bash
for d in $(ssh -i "$KEY" comma@<device-ip> 'ls -d /data/media/0/realdata/0000abcd--0123456789--*'); do
  mkdir -p ~/rlogs/my_drive/$(basename $d)
  scp -i "$KEY" "comma@<device-ip>:$d/rlog*" ~/rlogs/my_drive/$(basename $d)/
done
```

### (b) Download from comma connect

If your device is paired with [comma connect](https://connect.comma.ai) and online (Wi-Fi):

1. Open connect.comma.ai **in a browser on your computer**, pick the drive, open **Files**.
2. Under *All logs* click **Upload … logs** and wait in *View upload queue* until every rlog is uploaded
   (the device uploads them the next time it is online).
3. Download the `rlog.bz2` / `rlog.zst` files for the segments you want into one folder. Keep the segment
   number in each file name, e.g. `0000abcd--0123456789--5--rlog.zst`, so Driver-Mover can order them.

If the upload buttons are greyed out, check at [useradmin.comma.ai](https://useradmin.comma.ai) that the
device shows *Uploads Accepted*. useradmin also lists routes and download links for uploaded files.

### (c) Forks (BogPilot, FrogPilot, sunnypilot, …)

* Logs are stored in the same place (`/data/media/0/realdata/`) and the same way on forks.
* The bundled log schema is BogPilot's (a FrogPilot-based fork). It also reads stock openpilot and
  FrogPilot logs of a similar age. If you get schema errors with another fork, point Driver-Mover at that
  fork's own `cereal/` folder: `--schema-dir /path/to/that-fork/cereal`.
* Some forks rename or move messages (e.g. newer openpilot moved alert text from `controlsState` to
  `selfdriveState`); missing optional messages just leave that part of the HUD empty.

You can pass Driver-Mover the **route folder**, single **segment folders**, or individual
`rlog` / `rlog.zst` / `rlog.bz2` files.

---

## 3. Find interesting moments

On your own computer:

```bash
drivermover find ~/rlogs/my_drive
```

Example output:

```
 1. --start 302.0   (302.0-317.0 s; segment 5, 1.9 s in)  score 17.0
    1x Driver Distracted (orange), 2x Pay Attention, awareness dropped to 29%, head turned up to 71°
 2. --start 420.0   (420.0-435.0 s; segment 6, 59.9 s in)  score 3.6
    head turned up to 35°
```

Times are seconds from the start of the first segment you gave it. Options: `--top 10`, `--duration 20`.
Takes a few seconds for a 10-minute drive.

---

## 4. Render a video

**Overlay mode** (bundled mannequin photo):

```bash
drivermover render ~/rlogs/my_drive --start 302 --duration 15
```

**Realistic-head mode** (bundled mannequin keyframes are used automatically):

```bash
drivermover render ~/rlogs/my_drive --start 102 --duration 15 --realhead
```

* Leave out `--start` to auto-pick the best window from `find`.
* Output goes to `drivermover_out/` in the folder you ran the command from, e.g.
  `drivermover_out/dm_my_drive_302s.mp4`, plus a `_contact.png` (6 stills) and `_poster.png`
  (skip them with `--no-sheet`). Realistic-head mode also writes a `_keyframes.png` sheet. Choose the
  file yourself with `--out ~/Videos/drive.mp4`.
* Typical time for 15 s: overlay ≈25 s on 8 CPU cores (≈1 min on 4); realistic head ≈40 s (≈1.5–2 min);
  the moving hands add ≈15 s. `--workers 2` uses fewer cores.
* `--hands off` keeps the hands and wheel of the photo still (angle shown on the rim only); `--no-wheel`
  leaves the steering wheel out entirely.
* Videos are H.264 MP4, 30 fps, with container metadata stripped.

How the steering wheel works (v2, both modes, `--no-wheel` to switch it off):

* **Angle** – `carState.steeringAngleDeg`, the real wheel angle, so it includes your own steering. Multi-turn
  angles (e.g. −400° in a parking lot) are kept; frames between the 100 Hz samples are interpolated.
* **Who is steering** – from `carControl.latActive` and `carState.steeringPressed` (held 0.3 s so it doesn't
  flicker): **OPENPILOT** (green) = openpilot is steering, **OVERRIDE** (amber) = openpilot is steering and
  the driver is pushing the wheel, **DRIVER** (grey) = openpilot isn't steering.
* **Moving hands (`--hands on`, default, bundled mannequin)** – the photo is split once into layers
  (`tools/build_hands.py`, assets in `examples/mannequin/hands/`): a clean plate without arms or spokes
  (LaMa-inpainted, hidden rim arcs re-painted from the visible rim), the spokes, and the two arms cut out of the
  photo. Every frame they are stacked again: plate → arms (2-bone skinning: shoulder fixed, the elbow follows
  the wrist a little, the forearm stretches) → spokes turned by the angle in the wheel's own plane → rim →
  hands. Each hand is rigid, grips the rim and turns with it, tilting with the rim's tangent. Before a hand
  leaves its comfortable range it lets go and slides back along the rim (≈0.3–0.5 s, lifted a little) while
  the other hand keeps holding — **never both at once**; a hand that can't let go slips on the rim at its limit.
  When the wheel rests the hands drift back to their photo grips one at a time. At angle 0 with both hands home
  the frame is the original photo. Only the arm/wheel region (below the head zone) is touched, so the
  realistic head composes unchanged. Your own photo needs its own rig (masks/skeleton in
  `tools/build_hands.py`); without one the photo stays still and the angle is shown as below.
* **On the photo (`--hands off`)** – the photo itself is not changed. A faint ring on the rim in the who-is-steering colour
  and a solid tick marking the wheel's 12 o'clock turn with the angle; the hands are masked out, so the tick
  passes *behind* them. While
  openpilot steers, a white outlined ghost tick shows `carControl.actuators.steeringAngleDeg` (the angle it
  asked for). The panel at the bottom right shows the angle, left/right, turns, and a small wheel drawn **as
  the driver sees it** (a left turn turns it counter-clockwise).
* **Direction** – the driver camera faces the driver, so it sees the wheel from behind: a left turn turns the
  wheel **clockwise in the photo**. (`wheel.positive_angle_is_clockwise_in_image` flips it if your photo is
  taken from the driver's side.)
* **`--warp-spokes` (experimental, off by default, implies `--hands off`)** – the rim outline is calibrated as an ellipse; the spokes are cut out in the wheel's own
  plane (perspective undone, rotated, perspective redone) and turned. A spoke hidden in shadow is filled with
  a mirrored copy of a visible one; what a moved spoke uncovers is filled from the background next to it at the
  same distance from the hub. The rim is round, so it stays the original photo, and so do the hands (they
  stay in front of the wheel, pixel-for-pixel) and everything outside the rim — the realistic head is not
  affected. On the bundled mannequin this can cut through the hands and smear the rim, which is why it is off.

How realistic-head mode works:

1. **Align** – each keyframe is matched to the photo by background/body features (SIFT + RANSAC) so its
   head lands exactly where the photo's head is.
2. **Measure** – MediaPipe face landmarks give each keyframe's head yaw/pitch; a **mirrored** copy of each
   keyframe covers the other side.
3. **Schedule** – for every moment of the log the closest pose is chosen, with hysteresis and a minimum
   hold time so it doesn't flicker; a little rigid shift/roll follows the data in between.
4. **Composite** – 0.2 s cross-dissolves between unwarped keyframe heads, inside a protected head/neck
   mask. Looking forward = the original photo.

---

## 5. Use your own photo

1. **Anchor the face** (on your own computer):

   ```bash
   drivermover calibrate --image ~/Pictures/driver.jpg --auto
   ```

   This finds the face, writes `driver.json` next to the photo, and saves
   `drivermover_out/driver_calibration.png`. Open the PNG: the yellow cross should sit between the eyes,
   the circles on the eyes, the wireframe on the head, and the pink outline should cover head + neck but
   **not** the wheel or hands.

2. If it's off, set the values by hand (pixel coordinates, 0,0 = top-left) and check again:

   ```bash
   drivermover calibrate --image ~/Pictures/driver.jpg --eye-mid 812 590 --interocular 58 --yaw 12
   ```

   * `--eye-mid X Y` – point between the eyes
   * `--interocular PX` – distance between eye centres (scales everything)
   * `--yaw DEG` – how far the photo's head is already turned (+ = toward image-right); wireframe only

3. **Steering wheel**: `--auto` also looks for the wheel rim near where it sits relative to the face in the
   mannequin photo. In the preview the **cyan** ellipse should follow the rim's outer edge, the orange outlines
   are the spokes that will turn and the pink blobs are hands kept in front. If it's off, set the rim by hand
   (centre x y, full width, full height, tilt in degrees — the OpenCV ellipse convention):

   ```bash
   drivermover calibrate --image ~/Pictures/driver.jpg --wheel-ellipse 977 953 331 508 46.5
   ```

   Spoke positions (`wheel.spokes`, clock angles as the camera sees the wheel) can be edited in the JSON.
   Without a wheel calibration the video is rendered without the turning wheel.

4. Render: `drivermover render ~/rlogs/my_drive --image ~/Pictures/driver.jpg` (the `.json` beside the photo
   is loaded automatically, or pass `--config file.json`).

The HUD layout is designed for portrait photos (like the 1152×1728 example); other sizes work, centred.

---

## 6. Make keyframes for realistic-head mode

Use any image generator or face-rotation / photo-editing tool. Ask it to keep **the exact same photo —
same person, camera, framing, lighting, background, hands and wheel — and change only the head pose**.
Save the results (`.jpg` / `.png`, any names) in one folder.

| Pose | Purpose |
|---|---|
| your photo itself | “looking forward” – used automatically, don't add it |
| head turned ≈20–25° to one side | small glances, mirror checks |
| head turned ≈45–60° to the same side | big looks (side mirror, shoulder check) |
| head tilted down ≈15–25° | looking at the lap / phone / console |
| *optional:* the other side of each | otherwise mirrored copies are generated |
| *optional:* nearly frontal (≈0°) | helps when your photo's head is turned a lot |

Tips: same resolution/framing as the photo works best (keyframes are re-aligned, but enough of the
background and body must be visible to match); avoid changing hair or lighting; a real other-side keyframe
looks better than a mirrored one when the scene is lit from one side.

```bash
drivermover render ~/rlogs/my_drive --image ~/Pictures/driver.jpg --realhead --keyframes ~/Pictures/driver_keyframes
```

Check the `_keyframes.png` sheet: each keyframe is shown after alignment with its measured yaw/pitch and
whether it was used (one within 5° yaw / 4° pitch of another is skipped as a duplicate).

The bundled example lives in `src/drivermover/examples/mannequin/` (`photo.jpg`, `photo.json`,
`keyframes/head_down.jpg`, `turn_small.jpg`, `turn_large.jpg`, `near_front.jpg`).

---

## 7. Settings file reference

`<photo>.json` (written by `calibrate`; example: `src/drivermover/examples/mannequin/photo.json`). Missing
keys fall back to the defaults below.

| Key | Default | Meaning |
|---|---|---|
| `face.eye_mid` | – | `[x, y]` point between the eyes (pixels) |
| `face.interocular_px` | – | eye-centre distance; all offsets scale by `interocular_px / 56` |
| `face.apparent_yaw_deg` / `apparent_pitch_deg` | 12 / 0 | the photo's own head pose, for the wireframe |
| `data_mapping.positive_yaw_is_image_right` | true | flips the wireframe's turn direction (e.g. for a mirrored photo) |
| `data_mapping.positive_pitch_is_up` | true | flips the wireframe's nod direction |
| `hud.speed_units` | `mph` | or `kmh` |
| `hud.vignette` | true | darken the photo edges |
| `realhead.dissolve_s` | 0.2 | cross-dissolve length (s) |
| `realhead.min_hold_s` | 0.35 | minimum time on one keyframe |
| `realhead.hysteresis_deg` | 8 | how much better a new pose must fit before switching |
| `realhead.near_photo_yaw_deg` | 8 | keyframes this close to the photo's own pose are skipped (the photo is used instead) |
| `realhead.dedupe_yaw_deg` / `dedupe_pitch_deg` | 5 / 4 | keyframes closer than this count as duplicates |
| `realhead.down_keyframe_pitch_deg` | 5 | keyframes pitched down more than this are “head-down” |
| `realhead.head_down_start_deg` / `head_down_full_deg` | 8 / 22 | data pitch where head-down keyframes fade in / are full |
| `realhead.micro_shift_px` / `micro_roll_deg` | 4 / 2 | max rigid shift / roll between keyframes |
| `wheel.ellipse` | – | rim outline `[cx, cy, width, height, angle_deg]` (written by `calibrate`) |
| `wheel.up_dir` | `[0, -1]` | image direction of the wheel's 12 o'clock from its centre |
| `wheel.positive_angle_is_clockwise_in_image` | true | + steering angle (left) turns the photo's wheel clockwise (camera faces the driver) |
| `wheel.rim_inner` / `hub_radius` | 0.80 / 0.40 | rim inner edge and static hub, as a fraction of the rim radius |
| `wheel.spokes` | `[[270, 20], [190, 30]]` | spokes that turn: `[clock angle, half-width]` in degrees as the camera sees the wheel (driver's 3 o'clock = 270) |
| `wheel.mirror_spokes` | `[[270, 90]]` | spokes hidden in shadow, filled with a mirrored copy: `[visible, hidden]` |
| `wheel.gap_quantile` | 0.75 | inside a spoke wedge, pixels brighter than this quantile are see-through gaps and stay put |
| `wheel.hand_threshold` / `hand_min_area` | 55 / 400 | brightness / size of the hands kept in front of the wheel |
| `realhead.allow_polygon_rel` | (mannequin) | protected head/neck zone as offsets from `eye_mid` in reference pixels — edit if the pink outline in the calibration preview touches the wheel or hands |

The anchoring offsets (face box, head crop, protected zone) were derived on the mannequin photo (eye_mid
808, 592; interocular 56 px) and scale with `interocular_px`.

---

## 8. How the numbers are computed (assumptions)

* **Messages used**: `driverStateV2` (net face orientation/position, face/eye/blink/sunglasses/phone
  probabilities), `driverMonitoringState` (events, awareness, distracted flags), `controlsState` (alert text),
  `carState` (speed, steering angle, steering pressed/torque), `carControl` (openpilot steering active and its
  commanded wheel angle), `liveCalibration` (device mount angles). Nothing else is read.
* **Head angles** follow openpilot's `face_orientation_from_net`: the net's angles are corrected for where
  the face sits in the 1928×1208 driver-camera image (focal length 598 px) and for the device mount
  (`liveCalibration.rpyCalib`, per sample). + yaw = driver's right.
* **Straight ahead**: openpilot learns the driver's usual pose after ~60 s; until then it uses fixed
  “natural” offsets. The HUD shows whichever openpilot was using. The moving head is drawn relative to the
  driver's own forward pose (median of the 15 s before and after the window, excluding distracted moments),
  so the original photo = looking forward.
* **Left-hand drive** unless the log says right-hand drive (`isRHD`). The camera is assumed to face the
  driver from the windshield centre, so the driver turning to their right appears toward image-left.
* **Thresholds** shown (pose yaw ≈23–29°, pitch ≈26°, awareness 8/11 pre-alert, 6/11 prompt) are openpilot's
  driver-monitoring defaults.
* Roll is drawn clockwise-positive (not verifiable from the data). Gaze isn't logged separately; gaze lines
  follow head pose.
* The “Distracted” pill reflects openpilot's own pose/blink/phone flags; at low speed openpilot may flag a
  pose as distracted without lowering awareness or alerting (as in the example video).

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `drivermover: command not found` | `source .venv/bin/activate` in the project folder |
| `ffmpeg not found` | Linux `sudo apt install ffmpeg`, macOS `brew install ffmpeg` |
| `mediapipe could not start … libGLESv2` (Linux) | `sudo apt install libgles2 libegl1` |
| `pip install` fails building pycapnp / mediapipe | use Python 3.10–3.13 (`python3 --version`), `pip install --upgrade pip` |
| `no rlog / rlog.zst / rlog.bz2 files found` | point at the folder *containing* the `…--0`, `…--1` folders, or at the files; keep `--N` in file names |
| schema / “Not a valid capnp” errors | logs from a different fork: `--schema-dir /path/to/fork/cereal` |
| `no driverStateV2 / driverMonitoringState messages` | you gave it qlogs, or DM wasn't running — use rlogs |
| many `W0000 …` / `INFO: Created TensorFlow Lite …` lines | harmless MediaPipe messages |
| realistic head: “skipping …: too few matching features” | regenerate that keyframe closer to the original framing |
| realistic head looks pasted or turns the wrong way | check the `_keyframes.png` sheet; remove bad keyframes |
| “no steering-wheel calibration for this photo” | run `drivermover calibrate --image <photo> --auto` (or `--wheel-ellipse ...`) |
| wheel turns the wrong way | flip `wheel.positive_angle_is_clockwise_in_image` in the photo's JSON |
| hands look wrong on your own photo | `--hands off`, or build a rig for it: copy `tools/build_hands.py`, retrace the polygons/skeleton points |
| wheel looks smeared / background moves with it | check the calibration preview: the cyan ellipse must sit on the rim; narrow `wheel.spokes` |
| wireframe not on the face | re-run `drivermover calibrate` with `--eye-mid` / `--interocular` |

---

## 10. Project layout

```
Driver-Mover/
├── README.md  LICENSE  THIRD_PARTY_NOTICES.md  pyproject.toml
├── tests/                   pytest (python -m pytest tests; no rlogs needed)
├── tools/build_hands.py     rebuilds examples/mannequin/hands/ (optional LaMa inpainting)
├── docs/example_mirror_checks.{mp4,gif}   the example above
└── src/drivermover/
    ├── cli.py         `drivermover` command (find / render / calibrate)
    ├── logreader.py   finds segments, decompresses .zst/.bz2, reads with the bundled schemas
    ├── data.py        log messages → timelines (head angles, awareness, alerts, speed)
    ├── find.py        scores windows for `find` / auto-pick
    ├── config.py      photo settings (JSON) and face-anchor geometry
    ├── overlay.py     HUD + wireframe drawing, ffmpeg encoding
    ├── landmarks.py   MediaPipe face landmarks / head-angle measurement
    ├── realhead.py    keyframe alignment, mirroring, pose schedule, cross-dissolve, protected mask
    ├── wheel.py       steering wheel: rim ellipse, wheel-plane rotation, spokes, auto calibration
    ├── hands.py       moving hands: layered plate/arms/spokes/hands, 2-bone skinning, push-pull grip planner
    ├── schema/        cereal log schemas (MIT, comma.ai — via BogPilot)
    ├── fonts/         Inter (SIL OFL 1.1)
    ├── models/        MediaPipe face landmarker (Apache-2.0)
    └── examples/mannequin/  example photo, settings, keyframes and hands/ (layers + rig for --hands)
```

---

## 11. Licenses and credits

* Driver-Mover code: **MIT** (see `LICENSE`).
* `src/drivermover/schema/`: cereal log schemas from comma.ai openpilot (via the BogPilot fork),
  **MIT**, © comma.ai — `schema/LICENSE`; `include/c++.capnp` from Cap'n Proto, MIT.
* `src/drivermover/fonts/`: **Inter** by Rasmus Andersson, SIL Open Font License 1.1 — `fonts/OFL.txt`.
* `src/drivermover/models/face_landmarker.task`: **MediaPipe** Face Landmarker by Google, Apache-2.0.
* Example mannequin photo and keyframes: provided for this project.

Details in `THIRD_PARTY_NOTICES.md`. openpilot and comma are trademarks of comma.ai; this project is not
affiliated with comma.ai.
