"""Rolling skeleton window buffer + disk serialization (Milestone 1).

Maintains the last T=60 frames of `[13, 4]` landmarks and exports them as `.npz`
windows to ``data/sessions/``. Also provides the torso-normalization defined in
CLAUDE.md, applied defensively so missing landmarks never crash the pipeline.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from capture.pose_mediapipe import J, JOINT_NAMES, NUM_CHANNELS, NUM_JOINTS

WINDOW_SIZE = 60  # T: 1–3 second windows @ ~30fps (CLAUDE.md hard rule)
DEFAULT_SESSION_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"

# A landmark is considered usable for normalization geometry above this visibility.
_VIS_EPS = 1e-3
_SCALE_EPS = 1e-6


def normalize_window(window: np.ndarray) -> np.ndarray:
    """Torso-normalize a window of shape ``[T, J, 4]`` (CLAUDE.md conventions).

    Works for the 13-joint body skeleton and the extended body+hands skeleton alike —
    the torso anchors live in the first 13 (body) joints, and every joint (hands
    included) is expressed in the same body-relative frame::

        origin = midpoint(left_hip, right_hip)
        scale  = distance(left_shoulder, right_shoulder)
        normalized_joint_xyz = (joint_xyz - origin) / scale

    The visibility channel is preserved untouched. Frames whose torso anchors are
    missing/low-visibility (scale ~ 0) are left unnormalized rather than producing
    NaNs/Infs — graceful degradation, never a crash.
    """
    window = np.asarray(window, dtype=np.float32)
    if window.ndim != 3 or window.shape[2] != NUM_CHANNELS or window.shape[1] < NUM_JOINTS:
        raise ValueError(
            f"Expected window of shape [T, J>={NUM_JOINTS}, {NUM_CHANNELS}], got {window.shape}"
        )

    out = window.copy()
    lh, rh = J["left_hip"], J["right_hip"]
    ls, rs = J["left_shoulder"], J["right_shoulder"]

    for t in range(out.shape[0]):
        frame = out[t]
        # Require all four torso anchors to be visible to define the frame.
        if min(frame[lh, 3], frame[rh, 3], frame[ls, 3], frame[rs, 3]) <= _VIS_EPS:
            continue

        origin = (frame[lh, :3] + frame[rh, :3]) / 2.0
        scale = float(np.linalg.norm(frame[ls, :3] - frame[rs, :3]))
        if scale < _SCALE_EPS:
            continue

        # Only shift/scale joints that are present (visibility > 0); leave missing at 0.
        present = frame[:, 3] > _VIS_EPS
        frame[present, :3] = (frame[present, :3] - origin) / scale

    return out


class SkeletonBuffer:
    """A rolling buffer of the most recent ``maxlen`` skeleton frames."""

    def __init__(
        self,
        maxlen: int = WINDOW_SIZE,
        session_dir: str | Path = DEFAULT_SESSION_DIR,
        *,
        num_joints: int = NUM_JOINTS,
        joint_names: tuple[str, ...] = JOINT_NAMES,
    ) -> None:
        self.maxlen = maxlen
        self.session_dir = Path(session_dir)
        self.num_joints = num_joints
        self.joint_names = tuple(joint_names)
        self._frames: deque[np.ndarray] = deque(maxlen=maxlen)

    def append(self, landmarks: Optional[np.ndarray]) -> None:
        """Append one `[num_joints, 4]` frame. ``None`` becomes an all-missing frame."""
        if landmarks is None:
            landmarks = np.zeros((self.num_joints, NUM_CHANNELS), dtype=np.float32)
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.shape != (self.num_joints, NUM_CHANNELS):
            raise ValueError(
                f"Expected landmark frame [{self.num_joints}, {NUM_CHANNELS}], got {landmarks.shape}"
            )
        self._frames.append(landmarks.copy())

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def is_full(self) -> bool:
        return len(self._frames) == self.maxlen

    def window(self, *, normalize: bool = False) -> np.ndarray:
        """Return the current buffer as a `[T, 13, 4]` array (T = current length)."""
        if not self._frames:
            return np.zeros((0, self.num_joints, NUM_CHANNELS), dtype=np.float32)
        stacked = np.stack(list(self._frames), axis=0)
        return normalize_window(stacked) if normalize else stacked

    def clear(self) -> None:
        self._frames.clear()

    def save_window(
        self,
        *,
        fps: float = 0.0,
        normalize: bool = False,
        session_dir: Optional[str | Path] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[Path]:
        """Save the current window to ``data/sessions/`` as ``.npz``.

        The archive holds ``{timestamp, fps, landmarks: [T, 13, 4]}`` (CLAUDE.md spec),
        plus ``joint_names`` and ``normalized`` for self-describing replay. Returns the
        written path, or ``None`` if the buffer is empty.
        """
        if not self._frames:
            return None

        out_dir = Path(session_dir) if session_dir is not None else self.session_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = time.time() if timestamp is None else timestamp
        landmarks = self.window(normalize=normalize)

        # Filesystem-safe, sortable filename.
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
        path = out_dir / f"session_{stamp}_{int((ts % 1) * 1000):03d}.npz"

        np.savez_compressed(
            path,
            timestamp=np.float64(ts),
            fps=np.float32(fps),
            landmarks=landmarks.astype(np.float32),
            joint_names=np.array(self.joint_names, dtype=object),
            normalized=np.bool_(normalize),
        )
        return path
