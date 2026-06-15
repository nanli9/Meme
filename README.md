# meme-motion

A Mac-first prototype that reads body pose + facial expression, estimates expressive
intent, and recommends reaction memes. See [`CLAUDE.md`](./CLAUDE.md) for the full design.

> **Status: Sprint 1 / Milestone 1 only** — a real-time webcam skeleton debugger that
> logs pose windows to disk. No memes, no face, no ML, no ranking yet.

## Setup

```bash
uv venv --python 3.11
uv sync                       # installs deps from pyproject.toml / uv.lock
bash scripts/download_model.sh   # fetches pose "full" + hand landmarker models
```

The models in `data/models/` (`pose_landmarker_full.task`, `hand_landmarker.task`) are
git-ignored; the script above downloads them. Pass `lite`/`heavy`/`hands` to fetch a
single variant.

## Run the skeleton debugger

**Recommended — native window (runs at full camera speed, ~30–60 fps):**

```bash
uv run python -m capture.debug_view            # --device N, --normalize, --no-mirror, --no-hands
```

Keys: `s` = save the current 60-frame window · `q` / `ESC` = quit.
Fingers (MediaPipe Hands) are on by default; pass `--no-hands` for body-only.

**Optional — Streamlit GUI** (browser; lower fps because each frame is encoded and
shipped over the websocket — use the native window for smooth playback):

```bash
uv run streamlit run app/skeleton_debugger.py
```

Both show: the live feed with the skeleton overlay, live **FPS**, count of **visible
landmarks** (/55, or /13 body-only), current **buffer length** (/60), the detected
**gesture** (Milestone 2), plus a save action that writes the last 60 frames to
`data/sessions/*.npz`.

## Gestures (Milestone 2)

The debugger labels a broad **visible-reaction** category from the rolling window
(updated ~1×/sec, no flicker): `shrug`, `hype`, `arms_crossed`, `arms_wide`, `facepalm`,
`thinking`, `wave`, `clap`, `thumbs_up`, `pointing`, `open_palm`, or `neutral`. This is a
weak signal for later meme retrieval — not an emotion claim, not a meme ID.

Label / tune against saved windows offline (no camera):

```bash
uv run python scripts/label_session.py            # newest session
uv run python scripts/label_session.py --all      # every saved session
```

Rules live in `models/motion_rules.py`; features in `features/skeleton_features.py`.
Thresholds are hand-tuned — expect to refine them against your own saved poses.

### Evaluate against labeled datasets

Instead of posing everything by hand, run the rules over public labeled data and get a
confusion report (`evaluation/gesture_eval.py` holds the label maps):

```bash
# Finger gestures (thumbs_up / pointing / open_palm) — HaGRID images, auto-downloaded (~1GB, cached)
uv run python scripts/eval_hagrid.py --limit-per-class 150

# Body/motion gestures (wave / clap / arms_crossed / pointing) — NTU RGB+D 60 skeletons
# NTU is access-gated: request at https://rose1.ntu.edu.sg/dataset/actionRecognition/
uv run python scripts/eval_ntu.py --data-dir /path/to/nturgbd_skeletons
```

No public dataset covers `shrug`, `arms_wide`, `facepalm`, or `thinking` — tune those
from your own saved windows.

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

Body = 13 joints — `nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip, L/R knee, L/R
ankle` — each `[x, y, z, visibility]`. With fingers enabled (default) the skeleton is
**55 joints**: `[0:13]` body, `[13:34]` left hand (21), `[34:55]` right hand (21), so a
saved window is `[T=60, J=55, C=4]` (`J=13` in body-only mode). Hand points have no
native visibility, so they carry the hand's detection confidence. Windows can be
torso-normalized (origin = hip midpoint, scale = shoulder width) — hands included, in
the same body-relative frame. Each `.npz` stores `joint_names`, so replay handles both
13- and 55-joint windows.

> Note: the 55-joint skeleton with hands extends CLAUDE.md's documented 13-joint spec
> (added on request for finger detail). Body pose remains the primary signal.

## Saved `.npz` schema

| key          | shape / type        | meaning                                  |
|--------------|---------------------|------------------------------------------|
| `timestamp`  | float64             | wall-clock save time (epoch seconds)     |
| `fps`        | float32             | measured capture FPS at save time        |
| `landmarks`  | float32 `[T,J,4]`   | the window of `[x,y,z,visibility]` joints (J=55 with hands, 13 body-only) |
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
