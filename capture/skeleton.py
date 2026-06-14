"""Full skeleton = 13 body joints + 21 left-hand + 21 right-hand (fingers add-on).

Layout of the [55, 4] array:
    [ 0:13]  body   (see pose_mediapipe.JOINT_NAMES)
    [13:34]  left hand  (hands_mediapipe.HAND_LANDMARK_NAMES)
    [34:55]  right hand

Body pose stays the primary signal (CLAUDE.md). Hands were added on explicit request
for finger detail. Torso normalization still uses the body hip/shoulder anchors, so the
hands are expressed in the same body-relative frame.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from capture.hands_mediapipe import (
    HAND_EDGES,
    HAND_LANDMARK_NAMES,
    NUM_HAND_LANDMARKS,
    HandEstimator,
)
from capture.pose_mediapipe import (
    JOINT_NAMES as BODY_JOINT_NAMES,
    NUM_JOINTS as NUM_BODY_JOINTS,
    SKELETON_EDGES as BODY_EDGES,
    J as BODY_J,
    PoseEstimator,
)

LEFT_HAND_OFFSET = NUM_BODY_JOINTS               # 13
RIGHT_HAND_OFFSET = NUM_BODY_JOINTS + NUM_HAND_LANDMARKS  # 34
NUM_FULL_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_LANDMARKS  # 55
NUM_CHANNELS = 4

FULL_JOINT_NAMES: tuple[str, ...] = (
    *BODY_JOINT_NAMES,
    *(f"left_{n}" for n in HAND_LANDMARK_NAMES),
    *(f"right_{n}" for n in HAND_LANDMARK_NAMES),
)
assert len(FULL_JOINT_NAMES) == NUM_FULL_JOINTS

# Edges across the whole skeleton: body, each hand (offset), plus a link from each
# body wrist to its hand wrist so the fingers visually attach to the arm.
_left_hand_edges = tuple((a + LEFT_HAND_OFFSET, b + LEFT_HAND_OFFSET) for a, b in HAND_EDGES)
_right_hand_edges = tuple((a + RIGHT_HAND_OFFSET, b + RIGHT_HAND_OFFSET) for a, b in HAND_EDGES)
FULL_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    *BODY_EDGES,
    *_left_hand_edges,
    *_right_hand_edges,
    (BODY_J["left_wrist"], LEFT_HAND_OFFSET),    # arm -> left hand wrist
    (BODY_J["right_wrist"], RIGHT_HAND_OFFSET),  # arm -> right hand wrist
)


def assemble_skeleton(
    body: Optional[np.ndarray],
    left_hand: Optional[np.ndarray],
    right_hand: Optional[np.ndarray],
) -> np.ndarray:
    """Combine body `[13,4]` + two hands `[21,4]` into one `[55,4]` array.

    Any missing component is left as zeros (visibility 0) — graceful, never a crash.
    """
    out = np.zeros((NUM_FULL_JOINTS, NUM_CHANNELS), dtype=np.float32)
    if body is not None:
        out[:NUM_BODY_JOINTS] = body
    if left_hand is not None:
        out[LEFT_HAND_OFFSET:LEFT_HAND_OFFSET + NUM_HAND_LANDMARKS] = left_hand
    if right_hand is not None:
        out[RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + NUM_HAND_LANDMARKS] = right_hand
    return out


def draw_full_skeleton(
    frame_bgr: np.ndarray,
    skel: Optional[np.ndarray],
    *,
    visibility_threshold: float = 0.5,
    body_color: tuple[int, int, int] = (0, 255, 0),
    body_edge_color: tuple[int, int, int] = (0, 200, 255),
    hand_color: tuple[int, int, int] = (255, 0, 255),
    hand_edge_color: tuple[int, int, int] = (255, 200, 0),
) -> np.ndarray:
    """Draw a body+hands skeleton in place. Tolerates any joint count (skips out-of-range
    edges and missing/low-visibility points), so it also renders old 13-joint windows."""
    import cv2

    if skel is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    n = skel.shape[0]

    def px(j: int) -> Optional[tuple[int, int]]:
        if j >= n or skel[j, 3] < visibility_threshold:
            return None
        return int(skel[j, 0] * w), int(skel[j, 1] * h)

    for a, b in FULL_SKELETON_EDGES:
        if a >= n or b >= n:
            continue
        pa, pb = px(a), px(b)
        if pa is not None and pb is not None:
            is_hand = a >= NUM_BODY_JOINTS or b >= NUM_BODY_JOINTS
            cv2.line(frame_bgr, pa, pb, hand_edge_color if is_hand else body_edge_color, 2, cv2.LINE_AA)

    for j in range(n):
        p = px(j)
        if p is not None:
            is_hand = j >= NUM_BODY_JOINTS
            cv2.circle(frame_bgr, p, 3 if is_hand else 4, hand_color if is_hand else body_color, -1, cv2.LINE_AA)
    return frame_bgr


class FullPoseEstimator:
    """Runs body PoseLandmarker + HandLandmarker and returns a combined `[55, 4]` frame."""

    def __init__(self, *, visibility_threshold: float = 0.3, **pose_kwargs) -> None:
        self.visibility_threshold = visibility_threshold
        self.pose = PoseEstimator(visibility_threshold=visibility_threshold, **pose_kwargs)
        self.hands = HandEstimator()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        body = self.pose.process(frame_bgr, timestamp_ms)
        left, right = self.hands.process(frame_bgr, timestamp_ms)
        return assemble_skeleton(body, left, right)

    def close(self) -> None:
        self.pose.close()
        self.hands.close()

    def __enter__(self) -> "FullPoseEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
