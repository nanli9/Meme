# meme-motion

A Mac-first prototype that reads body pose + facial expression, estimates expressive
intent, and recommends reaction memes. See [`CLAUDE.md`](./CLAUDE.md) for the full design.

> **Status: Sprint 1 / Milestone 1 only** — a real-time webcam skeleton debugger that
> logs pose windows to disk. No memes, no face, no ML, no ranking yet.

## Setup

```bash
uv venv --python 3.11
uv sync                       # installs deps from pyproject.toml / uv.lock
bash scripts/download_model.sh   # fetch the MediaPipe Pose Landmarker model
```

The pose model (`data/models/pose_landmarker_lite.task`) is git-ignored; the script
above downloads it.

## Run the skeleton debugger

```bash
uv run streamlit run app/skeleton_debugger.py
```

Toggle **Run webcam** in the sidebar. You'll see:

- the live webcam feed with the 13-joint skeleton overlay,
- live **FPS**, count of **visible landmarks** (/13), and current **buffer length** (/60),
- a **💾 Save Window** button that writes the last 60 frames to `data/sessions/*.npz`.

## Replay a saved window offline

```bash
uv run python scripts/replay_session.py                 # newest session
uv run python scripts/replay_session.py path/to/x.npz   # a specific file
uv run python scripts/replay_session.py --no-display     # validate + summarize only
```

## Tests

```bash
uv run pytest -q
```

## Skeleton format

13 joints — `nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip, L/R knee, L/R ankle` —
each `[x, y, z, visibility]`. A saved window is `[T=60, J=13, C=4]`. Windows can be
torso-normalized (origin = hip midpoint, scale = shoulder width) per CLAUDE.md.

## Saved `.npz` schema

| key          | shape / type        | meaning                                  |
|--------------|---------------------|------------------------------------------|
| `timestamp`  | float64             | wall-clock save time (epoch seconds)     |
| `fps`        | float32             | measured capture FPS at save time        |
| `landmarks`  | float32 `[T,13,4]`  | the window of `[x,y,z,visibility]` joints |
| `joint_names`| object array (13)   | canonical joint order                    |
| `normalized` | bool                | whether torso-normalization was applied  |

## Sprint 1 acceptance criteria → where they live

1. Webcam opens successfully — `capture/webcam.py`
2. Skeleton landmarks drawn on the feed — `capture/pose_mediapipe.py::draw_skeleton`
3. Rolling 60-frame buffer — `features/skeleton_buffer.py::SkeletonBuffer`
4. Windows save to disk as `.npz` — `SkeletonBuffer.save_window`
5. Missing/low-visibility landmarks never crash — graceful handling throughout (+ tests)
6. Interactive FPS on Apple Silicon — MediaPipe runs on the Metal GPU
7. Saved windows replay offline — `scripts/replay_session.py`
