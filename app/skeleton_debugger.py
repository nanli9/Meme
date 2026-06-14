"""Real-time webcam skeleton debugger (Milestone 1).

Shows the live webcam feed with a skeleton overlay, live FPS, the count of visible
landmarks, the current rolling-buffer length, and a "Save Window" button that writes
the last 60 frames to ``data/sessions/`` as ``.npz``.

Run with::

    uv run streamlit run app/skeleton_debugger.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow `import capture...` / `import features...` when launched via Streamlit.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

from capture.pose_mediapipe import count_visible
from capture.skeleton import (
    FULL_JOINT_NAMES,
    NUM_FULL_JOINTS,
    FullPoseEstimator,
    draw_full_skeleton,
)
from capture.webcam import Webcam, WebcamError
from features.skeleton_buffer import WINDOW_SIZE, SkeletonBuffer

st.set_page_config(page_title="Skeleton Debugger", layout="wide")
st.title("🦴 Skeleton Debugger — Milestone 1")
st.caption("Live pose extraction + rolling 60-frame window logger. No memes, no face, no ML.")

# --- Sidebar controls ------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    device = st.number_input("Camera device index", min_value=0, max_value=8, value=0, step=1)
    vis_threshold = st.slider("Visibility threshold", 0.0, 1.0, 0.5, 0.05)
    normalize = st.checkbox("Torso-normalize saved windows", value=True)
    target_fps = st.slider("Target display FPS", 5, 60, 30, 1)
    run = st.toggle("Run webcam", value=False)


# --- Resource setup (persisted across reruns) ------------------------------
@st.cache_resource(show_spinner="Loading pose + hand models…")
def get_estimator(vis_threshold: float) -> FullPoseEstimator:
    return FullPoseEstimator(visibility_threshold=vis_threshold)


def get_webcam(device: int) -> Webcam:
    cam = st.session_state.get("_webcam")
    if cam is None or cam.device != device:
        if cam is not None:
            cam.release()
        cam = Webcam(device).open()
        st.session_state["_webcam"] = cam
    return cam


if "buffer" not in st.session_state:
    st.session_state.buffer = SkeletonBuffer(
        maxlen=WINDOW_SIZE, num_joints=NUM_FULL_JOINTS, joint_names=FULL_JOINT_NAMES
    )
if "saved" not in st.session_state:
    st.session_state.saved = []

buffer: SkeletonBuffer = st.session_state.buffer

# --- Layout placeholders ---------------------------------------------------
m1, m2, m3 = st.columns(3)
fps_box = m1.empty()
vis_box = m2.empty()
buf_box = m3.empty()
frame_box = st.empty()
status_box = st.empty()
saved_box = st.empty()


def render_metrics(fps: float, visible: int, buf_len: int) -> None:
    fps_box.metric("FPS", f"{fps:4.1f}")
    vis_box.metric("Visible landmarks", f"{visible} / {NUM_FULL_JOINTS}")
    buf_box.metric("Buffer length", f"{buf_len} / {WINDOW_SIZE}")


def render_saved() -> None:
    if st.session_state.saved:
        recent = st.session_state.saved[-5:]
        saved_box.success("Saved windows:\n" + "\n".join(f"- `{p}`" for p in recent))


def live_view(device: int, vis_threshold: float, normalize: bool, target_fps: int):
    """One captured/processed frame per fragment tick (timer set by caller)."""
    try:
        cam = get_webcam(device)
    except WebcamError as e:
        status_box.error(str(e))
        return
    estimator = get_estimator(vis_threshold)
    estimator.pose.visibility_threshold = vis_threshold

    ok, frame = cam.read()
    if not ok or frame is None:
        status_box.warning("Dropped a frame (camera returned nothing). Retrying…")
        render_metrics(cam.fps, 0, len(buffer))
        return

    landmarks = estimator.process(frame, int(time.time() * 1000))
    buffer.append(landmarks)

    overlay = draw_full_skeleton(frame.copy(), landmarks, visibility_threshold=vis_threshold)
    frame_box.image(overlay, channels="BGR", use_container_width=True)

    visible = count_visible(landmarks, vis_threshold)
    render_metrics(cam.fps, visible, len(buffer))

    # Save button lives inside the fragment so clicks are handled mid-stream.
    if st.button("💾 Save Window", type="primary", use_container_width=True):
        path = buffer.save_window(fps=cam.fps, normalize=normalize)
        if path is None:
            status_box.warning("Buffer is empty — nothing to save yet.")
        else:
            st.session_state.saved.append(path)
            status_box.success(f"Saved {len(buffer)}-frame window → `{path}`")
    render_saved()


if run:
    # Wrap as a self-refreshing fragment at the chosen tick interval, then call.
    fragment = st.fragment(run_every=1.0 / float(target_fps))(live_view)
    fragment(int(device), float(vis_threshold), bool(normalize), int(target_fps))
else:
    status_box.info("Toggle **Run webcam** in the sidebar to start.")
    render_metrics(0.0, 0, len(buffer))
    render_saved()
