"""Command line: drivermover find | render | calibrate"""
import argparse
import sys
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent / "examples" / "mannequin"


def _route(args):
  from .data import Route
  print(f"Reading rlogs from {', '.join(args.route)} ...", file=sys.stderr)
  r = Route(args.route, schema_dir=args.schema_dir)
  print(f"  {len(r.files)} segment file(s), {r.duration:.0f} s of driver monitoring, "
        f"{'right' if r.is_rhd else 'left'}-hand drive", file=sys.stderr)
  return r


def cmd_find(args):
  from .find import summary, windows
  r = _route(args)
  ws = windows(r, args.duration, args.top)
  if not ws:
    print("No driver-monitoring data worth showing was found."); return 0
  print(f"\nMost interesting {args.duration:.0f} s windows (times are seconds from the start of the route):\n")
  for i, w in enumerate(ws, 1):
    seg, off = r.segment_of(w["start"])
    print(f" {i}. --start {w['start']:.1f}   ({w['start']:.1f}-{w['end']:.1f} s; segment {seg}, {off:.1f} s in)  score {w['score']:.1f}")
    print(f"    {summary(w)}")
  print(f"\nMake a video of #1 with:\n  drivermover render {' '.join(args.route)} --start {ws[0]['start']:.1f}")
  return 0


def cmd_render(args):
  from . import config as C
  from .overlay import FPS, Renderer, encode
  t_begin = time.time()
  r = _route(args)
  if args.start is None:
    from .find import summary, windows
    ws = windows(r, args.duration, 1)
    if not ws:
      print("No interesting window found; use --start", file=sys.stderr); return 1
    args.start = ws[0]["start"]
    print(f"Auto-picked window {args.start:.1f}-{args.start + args.duration:.1f} s: {summary(ws[0])}", file=sys.stderr)
  if args.start + args.duration > r.duration + 1:
    print(f"Warning: the route only has {r.duration:.0f} s of data", file=sys.stderr)
  image = Path(args.image) if args.image not in (None, "example") else EXAMPLE_DIR / "photo.jpg"
  cfg = C.load(args.config, image)
  out = Path(args.out) if args.out else Path("drivermover_out") / f"dm_{Path(args.route[0]).resolve().name}_{args.start:.0f}s{'_realhead' if args.realhead else ''}.mp4"
  head = None
  if args.realhead:
    from PIL import Image
    import numpy as np
    from .config import Geometry
    from .realhead import HeadCompositor, Keyframes
    kdir = Path(args.keyframes) if args.keyframes else (EXAMPLE_DIR / "keyframes" if image.parent == EXAMPLE_DIR else None)
    if kdir is None:
      print("--realhead needs --keyframes <folder> for your own photo (see README)", file=sys.stderr); return 1
    print(f"Preparing realistic-head keyframes from {kdir} ...", file=sys.stderr)
    photo = np.asarray(Image.open(image).convert("RGB"))
    g = Geometry(cfg, (photo.shape[1], photo.shape[0]))
    kfs = Keyframes(photo, kdir, cfg, g)
    from . import landmarks
    landmarks.close()
    sheet = out.with_name(out.stem + "_keyframes.png"); out.parent.mkdir(parents=True, exist_ok=True)
    kfs.sheet(sheet, photo)
    print(f"  keyframe sheet: {sheet}", file=sys.stderr)
    rd0 = Renderer(r, image, cfg, args.start, args.duration)
    head = HeadCompositor(kfs, r, args.start, args.duration, photo, FPS, vignette=rd0.vig)
  rd = Renderer(r, image, cfg, args.start, args.duration, head=head)
  print(f"Rendering {rd.nf} frames ({args.duration:.0f} s at {FPS} fps) ...", file=sys.stderr)
  encode(rd, out, workers=args.workers)
  if not args.no_sheet:
    from PIL import Image, ImageDraw
    idx = [int(i * (rd.nf - 1) / 5) for i in range(6)]
    tiles = [Image.fromarray(rd.frame(i)) for i in idx]
    tw = 384; th = int(tw * rd.H / rd.W)
    cs = Image.new("RGB", (3 * tw + 40, 2 * th + 30), (10, 10, 12))
    for j, (i, im) in enumerate(zip(idx, tiles)):
      cs.paste(im.resize((tw, th), Image.LANCZOS), (10 + (j % 3) * (tw + 10), 10 + (j // 3) * (th + 10)))
    cs.save(out.with_name(out.stem + "_contact.png"))
    tiles[2].save(out.with_name(out.stem + "_poster.png"))
  size = out.stat().st_size / 1e6
  print(f"\nDone in {time.time() - t_begin:.0f} s:  {out}  ({size:.1f} MB)")
  return 0


def cmd_calibrate(args):
  from . import config as C
  from .overlay import preview
  image = Path(args.image) if args.image != "example" else EXAMPLE_DIR / "photo.jpg"
  cfg_path = Path(args.config) if args.config else image.with_suffix(".json")
  cfg = C.load(cfg_path if cfg_path.exists() else None)
  if args.auto or not cfg["face"].get("eye_mid"):
    import cv2
    import numpy as np
    from . import landmarks
    img = cv2.imread(str(image))
    lm = landmarks.detect(img, upscale=1.0)
    if lm is None:   # small face: search tiles
      H, W = img.shape[:2]; best = None
      for ty in np.linspace(0, H * 0.5, 3):
        for tx in np.linspace(0, W * 0.5, 3):
          tile = img[int(ty):int(ty + H * 0.5), int(tx):int(tx + W * 0.5)]
          d = landmarks.detect(tile, upscale=2.0)
          if d is not None:
            d["pts"] += [tx, ty]; best = d; break
        if best: break
      lm = best
    landmarks.close()
    if lm is None:
      print("Could not find a face automatically. Edit face.eye_mid / face.interocular_px in", cfg_path, "by hand.", file=sys.stderr)
      if not cfg["face"].get("eye_mid"):
        cfg["face"]["eye_mid"] = [img.shape[1] / 2, img.shape[0] / 3]; cfg["face"]["interocular_px"] = img.shape[1] / 20
    else:
      e = lm["pts"][[468, 473]]
      cfg["face"]["eye_mid"] = [round(float(e[:, 0].mean()), 1), round(float(e[:, 1].mean()), 1)]
      cfg["face"]["interocular_px"] = round(float(np.hypot(*(e[1] - e[0]))), 1)
      cfg["face"]["apparent_yaw_deg"] = round(lm["yaw"], 1)
      print(f"Found face: eye_mid {cfg['face']['eye_mid']}, interocular {cfg['face']['interocular_px']} px, head yaw {lm['yaw']:+.1f} deg", file=sys.stderr)
  for k, v in (("eye_mid", args.eye_mid), ("interocular_px", args.interocular), ("apparent_yaw_deg", args.yaw), ("apparent_pitch_deg", args.pitch)):
    if v is not None:
      cfg["face"][k] = v
  if args.config or args.auto or args.eye_mid or args.interocular or args.yaw is not None or args.pitch is not None or not cfg_path.exists():
    if image.parent == EXAMPLE_DIR and not args.config:
      print("(not overwriting the bundled example settings; pass --config my.json to save)", file=sys.stderr)
    else:
      C.save(cfg, cfg_path); print(f"Saved settings to {cfg_path}")
  out = Path(args.preview) if args.preview else Path("drivermover_out") / (image.stem + "_calibration.png")
  out.parent.mkdir(parents=True, exist_ok=True)
  preview(image, cfg, out)
  print(f"Preview with the anchor drawn: {out}")
  return 0


def main(argv=None):
  p = argparse.ArgumentParser(prog="drivermover", description="Driver-monitoring overlay videos from openpilot/BogPilot rlogs (runs on your computer).")
  sub = p.add_subparsers(dest="cmd", required=True)
  def route_args(q):
    q.add_argument("route", nargs="+", help="route folder (with <route>--N segment folders) or rlog / rlog.zst / rlog.bz2 files")
    q.add_argument("--duration", type=float, default=15.0, help="window length in seconds (default 15)")
    q.add_argument("--schema-dir", help="use another fork's cereal folder instead of the bundled BogPilot schema")
  q = sub.add_parser("find", help="list the most interesting driver-monitoring moments"); route_args(q)
  q.add_argument("--top", type=int, default=5)
  q.set_defaults(fn=cmd_find)
  q = sub.add_parser("render", help="make the overlay video"); route_args(q)
  q.add_argument("--image", default="example", help="photo to draw on (default: bundled mannequin photo)")
  q.add_argument("--config", help="face-anchor settings JSON (default: <photo>.json next to the photo)")
  q.add_argument("--start", type=float, help="window start in seconds from route start (default: best window)")
  q.add_argument("--out", help="output .mp4 (default: drivermover_out/...)")
  q.add_argument("--realhead", action="store_true", help="move a realistic head using keyframe head images")
  q.add_argument("--keyframes", help="folder of keyframe images (default with the example photo: bundled keyframes)")
  q.add_argument("--workers", type=int, help="CPU processes (default: all cores)")
  q.add_argument("--no-sheet", action="store_true", help="skip the contact sheet / poster PNGs")
  q.set_defaults(fn=cmd_render)
  q = sub.add_parser("calibrate", help="set / check the face anchor for a photo and write a preview PNG")
  q.add_argument("--image", required=True, help="your photo (or 'example')")
  q.add_argument("--config", help="settings file to write (default: <photo>.json)")
  q.add_argument("--auto", action="store_true", help="detect the face automatically")
  q.add_argument("--eye-mid", type=float, nargs=2, metavar=("X", "Y"))
  q.add_argument("--interocular", type=float, metavar="PX")
  q.add_argument("--yaw", type=float, help="how far the photo's head is already turned (+ = toward image-right)")
  q.add_argument("--pitch", type=float)
  q.add_argument("--preview", help="preview PNG path")
  q.set_defaults(fn=cmd_calibrate)
  args = p.parse_args(argv)
  try:
    return args.fn(args)
  except (FileNotFoundError, RuntimeError, ValueError) as e:
    print(f"error: {e}", file=sys.stderr); return 1


if __name__ == "__main__":
  sys.exit(main())
