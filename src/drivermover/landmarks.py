"""Face landmarks + head angles with MediaPipe Face Landmarker (CPU). Model file is bundled in models/."""
import atexit
import math
import os
from pathlib import Path

import numpy as np

MODEL = Path(__file__).parent / "models" / "face_landmarker.task"
_DET = None


def _detector():
  global _DET
  if _DET is None:
    os.environ.setdefault("GLOG_minloglevel", "2")       # keep MediaPipe's C++ logging quiet
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
      import mediapipe as mp
      from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as e:
      raise RuntimeError("mediapipe is needed for this step:  pip install mediapipe") from e
    try:
      opt = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)), output_facial_transformation_matrixes=True,
                                         num_faces=1, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2)
      _DET = (mp, vision.FaceLandmarker.create_from_options(opt))
    except OSError as e:
      raise RuntimeError(f"mediapipe could not start ({e}). On Ubuntu/Debian run:  sudo apt install libgles2 libegl1") from e
  return _DET


def detect(bgr, upscale=2.0):
  """Returns dict(pts=Nx2 in input pixels, yaw, pitch, roll) or None. yaw + = face turned toward image-right."""
  import cv2
  mp, det = _detector()
  big = cv2.resize(np.clip(bgr, 0, 255).astype(np.uint8), None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
  r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))))
  if not r.face_landmarks:
    return None
  pts = np.array([[p.x * big.shape[1] / upscale, p.y * big.shape[0] / upscale] for p in r.face_landmarks[0]], float)
  R = np.array(r.facial_transformation_matrixes[0])[:3, :3]
  yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])))
  pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
  roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
  return dict(pts=pts, yaw=yaw, pitch=pitch, roll=roll)


def close():
  """Release the landmarker (call before forking render workers / at exit)."""
  global _DET
  if _DET is not None:
    try:
      _DET[1].close()
    except Exception:
      pass
    _DET = None


atexit.register(close)
