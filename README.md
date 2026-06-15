# meme-motion

A Mac-first prototype that reads body pose + facial expression, estimates expressive
intent, and recommends reaction memes. See [`CLAUDE.md`](./CLAUDE.md) for the full design.

> **Status: Milestones 1–4 done.** Skeleton debugger + window logger (M1), skeleton
> features + rule-based gestures with a learned hand classifier (M2), the meme retrieval
> DB — subset load → OpenCLIP embeddings → auto-labeling → vector index → retrieval (M3),
> and the first end-to-end demo: pose → gesture → intent → ranked meme recommendations
> (M4). Next: M5 adds facial expression as a modifier. No face, no *learned* ranking yet.

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
`thinking`, `wave`, `clap`, `thumbs_up`, `pointing`, `middle_finger`, `pinky`, `peace`,
`open_palm`, or `neutral`. This is a weak signal for later meme retrieval — not an emotion
claim, not a meme ID. (Context-aware: an index finger at your own head reads as
`thinking`; a hand moving near your face reads as `wave`.)

Label / tune against saved windows offline (no camera):

```bash
uv run python scripts/label_session.py            # newest session
uv run python scripts/label_session.py --all      # every saved session
```

Rules live in `models/motion_rules.py`; features in `features/skeleton_features.py`.
Finger gestures use a **learned classifier** (`models/hand_classifier.py`) when a trained
model is present, falling back to rules otherwise. Train it on HaGRID (~92% held-out):

```bash
uv run python scripts/train_hand_classifier.py   # writes data/models/hand_gesture_clf.joblib
```

Body/motion gestures stay rule-based and are hand-tuned against your own saved windows.

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

## Meme database & retrieval (Milestone 3)

Build the retrieval DB end to end — pulls a small Hugging Face subset, embeds the images
with OpenCLIP, auto-labels reaction tags (CLIP zero-shot), and writes SQLite metadata +
a vector index. First run downloads the CLIP weights (~600 MB) and the dataset.

```bash
uv run python -m meme_db.build_index                  # default: MemeCap, ~5.8k memes
uv run python -m meme_db.build_index --source memecap,templates,not-lain   # combine -> ~7k
uv run python -m meme_db.build_index --source not-lain --limit 200         # small/fast
```

Source aliases: `memecap` (~5.8k, default), `memetion` (~7k, image-only), `templates`
(~1k named formats), `not-lain` (~300). Comma-separate to combine; `--limit 0` = whole
dataset, or set a per-source cap. Any other HF dataset id works too.

Then query it with an expressive-intent phrase (top-5 cosine, joined to metadata):

```bash
uv run python -m meme_db.retrieve "facepalm reaction"
uv run python -m meme_db.retrieve "hype celebration" -k 8
```

Artifacts (all git-ignored): images in `data/memes/`, metadata in `data/labels/memes.db`,
vectors in `data/embeddings/memes.npy`. Vector search uses **FAISS when available and an
exact NumPy cosine search otherwise** — the pipeline runs either way (faiss-cpu can be
finicky under uv on Mac). This is pure similarity + metadata; the scoring/diversity/safety
ranking lands in Milestone 4.

## Recommend memes from your pose (Milestone 4)

The first end-to-end demo: your body gesture → an intent query → retrieval → the hand-tuned
ranking formula → top-5 memes. Build the meme DB first (above), then:

```bash
# Live: webcam + skeleton on the left, top-5 recommended memes on the right.
uv run python scripts/recommend_demo.py            # --device N, --no-mirror, -k 5

# Camera-free: recommend for a saved .npz window (prints the ranking breakdown).
uv run python scripts/recommend_session.py --all   # or a path; --show opens the images
```

Ranking (in `models/fusion_ranker.py`): `0.50*CLIP + 0.25*tag_match + 0.15*intensity
+ 0.10*diversity − 0.20*recent_duplicate`, with flagged memes vetoed. Holding the same
gesture rotates through different memes (diversity + recent-duplicate memory). Weights are
hand-tuned now; Milestone 7 learns them from feedback.

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
