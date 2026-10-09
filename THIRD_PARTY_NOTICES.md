# Third-party files bundled with Driver-Mover

| What | Where | License |
|---|---|---|
| cereal log schemas (`log.capnp`, `car.capnp`, `custom.capnp`, `legacy.capnp`) copied from the BogPilot repository (a fork of comma.ai openpilot) | `src/drivermover/schema/` | MIT, Copyright (c) 2018, Comma.ai, Inc. — full text in `src/drivermover/schema/LICENSE` |
| `c++.capnp` (annotation file imported by the schemas) from Cap'n Proto | `src/drivermover/schema/include/` | MIT, Copyright (c) 2013-2014 Sandstorm Development Group, Inc. and contributors |
| Inter typeface (Regular, Medium, SemiBold, Bold) by Rasmus Andersson | `src/drivermover/fonts/` | SIL Open Font License 1.1 — full text in `src/drivermover/fonts/OFL.txt` |
| MediaPipe Face Landmarker model (`face_landmarker.task`) by Google | `src/drivermover/models/` | Apache License 2.0 — https://www.apache.org/licenses/LICENSE-2.0 (model card: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) |
| Example mannequin photo and keyframe head images | `src/drivermover/examples/mannequin/` | Provided by the project owner for this example; metadata stripped |

Python dependencies (numpy, opencv-python, pillow, pycapnp, zstandard, mediapipe) are installed by pip
from PyPI and keep their own licenses.
