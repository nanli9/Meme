"""OpenCV webcam capture with graceful device errors (Milestone 1)."""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np


class WebcamError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


class Webcam:
    """Thin wrapper around ``cv2.VideoCapture`` with friendly errors and FPS tracking.

    Usage::

        with Webcam(0) as cam:
            ok, frame = cam.read()
    """

    def __init__(
        self,
        device: int = 0,
        *,
        width: int = 1280,
        height: int = 720,
        warmup_frames: int = 5,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self._warmup_frames = warmup_frames
        self._cap: Optional[cv2.VideoCapture] = None

        # Rolling FPS estimate.
        self._last_t: Optional[float] = None
        self._fps_ema: float = 0.0

    def open(self) -> "Webcam":
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            raise WebcamError(
                f"Could not open webcam device {self.device!r}. "
                "Is another app using the camera, or is camera permission denied? "
                "On macOS, grant camera access to your terminal/IDE in "
                "System Settings > Privacy & Security > Camera."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap

        # Discard the first few frames; some cameras return black/garbage on warmup.
        for _ in range(self._warmup_frames):
            cap.read()
        return self

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read one BGR frame. Returns ``(ok, frame)``; never raises on a dropped frame."""
        if self._cap is None:
            raise WebcamError("Webcam not opened. Call open() or use as a context manager.")
        ok, frame = self._cap.read()
        if ok and frame is not None:
            self._update_fps()
        return ok, (frame if ok else None)

    def _update_fps(self) -> None:
        now = time.perf_counter()
        if self._last_t is not None:
            dt = now - self._last_t
            if dt > 0:
                inst = 1.0 / dt
                # Exponential moving average smooths the per-frame jitter.
                self._fps_ema = inst if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * inst
        self._last_t = now

    @property
    def fps(self) -> float:
        return self._fps_ema

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Webcam":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()
