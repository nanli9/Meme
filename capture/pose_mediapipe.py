"""MediaPipe Pose Landmarker wrapper (Milestone 1).

Extracts the 13 joints defined in CLAUDE.md from each frame as `[x, y, z, visibility]`.
The result-parsing logic is split into a pure function (`landmarks_from_result`) so it
can be unit-tested without a camera or GPU.

We use the modern MediaPipe Tasks `PoseLandmarker` (NOT the legacy `mp.solutions.pose`,
and NOT OpenPose — see CLAUDE.md tech stack rules).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Skeleton definition (CLAUDE.md "Skeleton conventions")
# ---------------------------------------------------------------------------

# The 13 joints, in canonical order. Index = our joint index J (0..12).
JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
NUM_JOINTS = len(JOINT_NAMES)  # 13
NUM_CHANNELS = 4  # x, y, z, visibility

# Map each canonical joint to its index in MediaPipe BlazePose's 33-landmark model.
MP_LANDMARK_INDICES: tuple[int, ...] = (
    0,   # nose
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
)

# Indices into our 13-joint array, by name, for convenience.
J = {name: i for i, name in enumerate(JOINT_NAMES)}

# Bones to draw, as pairs of indices into the 13-joint array.
SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (J["left_shoulder"], J["right_shoulder"]),
    (J["left_shoulder"], J["left_elbow"]),
    (J["left_elbow"], J["left_wrist"]),
    (J["right_shoulder"], J["right_elbow"]),
    (J["right_elbow"], J["right_wrist"]),
    (J["left_shoulder"], J["left_hip"]),
    (J["right_shoulder"], J["right_hip"]),
    (J["left_hip"], J["right_hip"]),
    (J["left_hip"], J["left_knee"]),
    (J["left_knee"], J["left_ankle"]),
    (J["right_hip"], J["right_knee"]),
    (J["right_knee"], J["right_ankle"]),
    (J["nose"], J["left_shoulder"]),
    (J["nose"], J["right_shoulder"]),
)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
# Prefer the higher-accuracy "full" model; fall back to "lite" if only that is present.
DEFAULT_MODEL_PATH = (
    _MODELS_DIR / "pose_landmarker_full.task"
    if (_MODELS_DIR / "pose_landmarker_full.task").exists()
    else _MODELS_DIR / "pose_landmarker_lite.task"
)


def landmarks_from_result(result, visibility_threshold: float = 0.0) -> Optional[np.ndarray]:
    """Convert a PoseLandmarkerResult into a `[13, 4]` float32 array.

    Returns ``None`` when no pose was detected. Joints whose visibility is below
    ``visibility_threshold`` are zeroed-out with visibility 0 so downstream code can
    treat them as "missing" without crashing (CLAUDE.md: handle missing landmarks
    gracefully).
    """
    if result is None or not getattr(result, "pose_landmarks", None):
        return None

    # Use the first detected pose only (single-user debugger).
    pose = result.pose_landmarks[0]

    out = np.zeros((NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
    for j, mp_idx in enumerate(MP_LANDMARK_INDICES):
        if mp_idx >= len(pose):
            # Defensive: model returned fewer landmarks than expected.
            continue
        lm = pose[mp_idx]
        vis = float(getattr(lm, "visibility", 0.0) or 0.0)
        if vis < visibility_threshold:
            # Keep as missing (already zeros, visibility 0).
            continue
        out[j, 0] = float(lm.x)
        out[j, 1] = float(lm.y)
        out[j, 2] = float(lm.z)
        out[j, 3] = vis
    return out


def count_visible(landmarks: Optional[np.ndarray], visibility_threshold: float = 0.5) -> int:
    """Number of joints whose visibility exceeds the threshold (0 if no detection)."""
    if landmarks is None:
        return 0
    return int(np.count_nonzero(landmarks[:, 3] >= visibility_threshold))


def draw_skeleton(
    frame_bgr: np.ndarray,
    landmarks: Optional[np.ndarray],
    *,
    visibility_threshold: float = 0.5,
    point_color: tuple[int, int, int] = (0, 255, 0),
    edge_color: tuple[int, int, int] = (0, 200, 255),
) -> np.ndarray:
    """Draw the 13-joint skeleton onto a BGR frame in place. Coords are normalized [0,1].

    Missing/low-visibility joints are simply skipped — never an error.
    """
    import cv2  # local import: keeps this module usable without OpenCV present

    if landmarks is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]

    def px(j: int) -> Optional[tuple[int, int]]:
        if landmarks[j, 3] < visibility_threshold:
            return None
        return int(landmarks[j, 0] * w), int(landmarks[j, 1] * h)

    for a, b in SKELETON_EDGES:
        pa, pb = px(a), px(b)
        if pa is not None and pb is not None:
            cv2.line(frame_bgr, pa, pb, edge_color, 2, cv2.LINE_AA)

    for j in range(NUM_JOINTS):
        p = px(j)
        if p is not None:
            cv2.circle(frame_bgr, p, 4, point_color, -1, cv2.LINE_AA)

    return frame_bgr


class PoseEstimator:
    """Per-frame pose extraction in VIDEO running mode.

    Usage::

        est = PoseEstimator()
        landmarks = est.process(frame_bgr, timestamp_ms)  # [13, 4] or None
        est.close()
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        visibility_threshold: float = 0.3,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. Download it with:\n"
                "  bash scripts/download_model.sh"
            )
        self.visibility_threshold = visibility_threshold

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        """Run the landmarker on one BGR frame. Returns `[13, 4]` or None."""
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB without an OpenCV dependency here
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return landmarks_from_result(result, self.visibility_threshold)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
