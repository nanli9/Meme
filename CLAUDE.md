# CLAUDE.md — meme-motion

Project memory for Claude Code. Read this at the start of every session. Keep edits high-signal.

## What we're building

A Mac-first prototype that reads body pose + facial expression, estimates **expressive intent** (a weak signal), and recommends/retrieves reaction memes that fit the moment.

The pipeline is:

```
skeleton motion + facial expression + optional context
    -> expressive intent (weak signal)
    -> meme retrieval + ranking
```

It is NOT `skeleton -> emotion -> meme`. That framing is brittle and forbidden.

## Why this product exists (design principles)

1. **Motion is a recall key text can't match.** Sometimes the user only remembers the *body* of a reaction (the lean-back, the slow blink, the arms-out shrug) and has no word to type. Gesture beats the search box precisely in that slice. Optimize for that slice; don't try to beat search everywhere.
2. **All signals are weak. Fuse, never trust one.** Skeleton drives broad reaction categories. Face is a modifier. Context disambiguates. The meme database + ranking is the real moat — treat it as the hardest, most important part.
3. **North-star vision (NOT in MVP): pose-driven meme generation.** Eventually the user's skeleton drives a character they pick (ControlNet-style pose-conditioned image generation). Do not build this yet, but keep the skeleton representation clean, normalized, and serializable so it can be used as a ControlNet control signal later. Never make an architecture choice that walls this off.

## Hard product rules

- Never claim "the user is angry/sad/depressed." Say "the visible expression resembles frustration/amusement/confusion." Estimate visible expression, never mental state.
- Use 1–3 second windows (T=60 frames @ ~30fps), never single frames.
- Update recommendations at most once every 1–2 seconds. No per-frame flicker.
- Same gesture must not always return the same meme (diversity + recent-duplicate penalty).
- No reading of messages, window titles, or screen content without explicit user opt-in.

## Current focus

Milestones 1–4 are DONE and verified. M1 = skeleton debugger + window logger; M2 =
skeleton features + rule-based gestures (+ a learned hand classifier brought forward);
M3 = meme DB (subset load → CLIP embeddings → auto-labeling → vector index → retrieval);
M4 = skeleton → meme baseline (gesture → intent → retrieve → rank → first live demo).
Now on **Milestone 5**: face expression module — a *modifier* on the skeleton signal,
never ground truth, never dominating skeleton (CLAUDE.md rules). Do not skip ahead.

## Tech stack

- Python 3.11/3.12, managed with `uv`.
- Pose: MediaPipe Pose Landmarker + OpenCV webcam capture + NumPy. Do NOT use OpenPose.
- Face (Milestone 5+): MediaPipe Face Landmarker first; ARKit blendshapes over WebSocket later.
- Memes: OpenCLIP for embeddings, FAISS (faiss-cpu) for vector search, SQLite for metadata.
- UI: Streamlit for the MVP. Do NOT start with Electron/Tauri/React or mobile.
- Apple Silicon preferred. If `faiss-cpu` fails under uv on Mac, fall back to conda for FAISS only.

## Repo layout

```
meme-motion/
  app/            streamlit_app.py, skeleton_debugger.py
  capture/        webcam.py, pose_mediapipe.py, face_mediapipe.py, arkit_receiver.py
  features/       skeleton_features.py, skeleton_buffer.py, face_features.py, temporal_smoothing.py
  models/         motion_rules.py, motion_encoder.py, face_encoder.py, fusion_ranker.py
  meme_db/        build_index.py, load_dataset.py, auto_label_memes.py, embed_memes.py, retrieve.py, schema.py
  data/           memes/, labels/, embeddings/, sessions/
  tests/          test_pose.py, test_features.py, test_retrieval.py
```

## Skeleton conventions

- 13 joints: nose, L/R shoulder, L/R elbow, L/R wrist, L/R hip, L/R knee, L/R ankle.
- Per landmark: `[x, y, z, visibility]`. Window shape: `[T=60, J=13, C=4]`.
- Normalize every window relative to torso before use:
  - `origin = midpoint(left_hip, right_hip)`
  - `scale = distance(left_shoulder, right_shoulder)`
  - `normalized_joint = (joint - origin) / scale`
- Handle missing/low-visibility landmarks gracefully — never crash.
- **Hands extension (added on request):** an optional MediaPipe Hands layer appends 21
  left + 21 right hand landmarks after the 13 body joints → a 55-joint skeleton
  (`[0:13]` body, `[13:34]` left hand, `[34:55]` right hand). Body pose stays primary;
  hands are in the same torso-normalized frame. Body-only mode (J=13) is still supported.
  Hand points use the hand's detection confidence as their visibility channel.

## Gestures (Milestone 2)

- Rule-based gestures map a normalized window → a broad **visible reaction** category.
  This is a WEAK signal fed to meme retrieval (M4) — never an emotion claim, never a
  direct meme ID.
- Vocabulary: `shrug`, `hype`, `arms_crossed`, `arms_wide`, `facepalm`, `thinking`,
  `wave`, `clap`, and finger gestures `thumbs_up`, `pointing`, `middle_finger`, `pinky`,
  `peace`, `open_palm`, + `neutral` fallback. Finger gestures use the 55-joint hands;
  each single-finger gesture needs its finger extended and the others curled. Context:
  an index finger AT your own head reads as `thinking` (not `pointing`); a moving hand
  near the face is `wave` (not `thinking`).
- Features in `features/skeleton_features.py`; hand-tuned rules + `GestureEstimate` +
  `GestureStabilizer` in `models/motion_rules.py`.
- Finger gestures (thumbs_up/pointing/open_palm/peace) now use a **learned classifier**
  (`models/hand_classifier.py`, small MLP on normalized MediaPipe hand landmarks, trained
  on HaGRID via `scripts/train_hand_classifier.py` — ~92% held-out). Rules are the
  fallback when no model file is present; body/motion gestures + middle_finger/pinky stay
  rule-based. This is Milestone 6 brought forward for hands.
- Cadence: gestures are computed over the 60-frame window and the displayed label updates
  at most once every 1–2s with hysteresis (no per-frame flicker).
- Labeled eval: `scripts/eval_hagrid.py` (HaGRID images → finger gestures) and
  `scripts/eval_ntu.py` (NTU RGB+D skeletons, access-gated → body gestures); label maps in
  `evaluation/gesture_eval.py`. No public set covers `shrug`/`arms_wide`/`facepalm`/
  `thinking` — tune those from self-recorded windows. Visually verify any saved window
  with `scripts/replay_session.py` (gesture overlaid on the replayed skeleton).

## Meme sourcing (read before building the DB)

- **Bootstrap from existing labeled datasets, not a scraper.** Pull a SUBSET (~200–300) of a Hugging Face dataset to prove the pipeline runs end to end. Good candidates: `not-lain/meme-dataset` (image+text, 300 rows), `MemeCap` (~6.3K with post title + captions + visual metaphors), `harpreetsahota/memes-dataset`.
- A random subset validates plumbing, NOT funniness. Curate for quality only after the pipeline works, then scale.
- Reaction-GIF APIs have shifted (2026): **Tenor's public API shuts down June 30, 2026 — do not build on it.** GIPHY works but is paid for production (free beta keys are rate-limited to ~100 calls/hour). **Klipy** is the free Tenor-compatible alternative and has a dedicated Meme API.
- Reddit (PRAW) is viable: post titles from r/reactiongifs etc. double as free weak labels.
- Legal reality: most memes are copyrighted (movie stills, celebrity photos, characters). Fine for a private/research MVP; flag `copyrighted_character` / `private_person` in metadata. The GIF-API route is the cleanest if this ever ships.

## Meme DB / retrieval (Milestone 3 — built)

The retrieval infra in `meme_db/`. One command builds everything (default: MemeCap ~5.8k):
`uv run python -m meme_db.build_index` (or `--source memecap,templates --limit 0` to
combine; `--source not-lain --limit 200` for a fast small build). Embedding is chunked
from disk (`MemeEmbedder.embed_image_paths`) so large builds don't hold thousands of
images open at once.

- `load_dataset.py` — pulls HF meme source(s), saves images to `data/memes/`. **Default is
  now MemeCap** (`Leonardo6/memecap`, ~5.8k real memes with rich captions). Handles
  generic `image`/`text` columns, an image under `local_path`, and MemeCap's nested
  `messages`+`images` schema. Aliases: `memecap`, `memetion` (~7k, image-only),
  `templates` (~1k named formats), `not-lain` (~300). `--source a,b,c` combines sources
  (~14k total); `--limit 0` = whole dataset. Avoid the `hateful_memes` sets (off-theme).
- `embed_memes.py` — `MemeEmbedder` (OpenCLIP `ViT-B-32 / laion2b_s34b_b79k`, image+text,
  L2-normalized so inner product = cosine) and `VectorIndex`. **`VectorIndex` uses FAISS
  when importable and falls back to an identical exact NumPy cosine search otherwise**
  (CLAUDE.md flags faiss-cpu can fail under uv on Mac; for a few-hundred-meme bootstrap
  brute force is instant). Source of truth on disk is `data/embeddings/memes.npy`.
- `auto_label_memes.py` — CLIP **zero-shot** reaction tags + an `intensity` proxy over
  `TAG_VOCABULARY` (no API key; reuses the loaded CLIP model). These feed the M4 ranking
  `tag_match`/`intensity_match`. LLM labeling is the documented offline upgrade path.
- `schema.py` — SQLite (`data/labels/memes.db`). Integer `id` == embedding row, so a
  search hit maps straight to metadata. Stores `tags`, `intensity`, and the
  `copyrighted_character`/`private_person` safety flags (M4 wires the veto).
- `retrieve.py` — `retrieve(text, k)` (embeds an intent phrase) and `retrieve_by_vector(vec, k)`
  (for M4's fused intent vector). Pure similarity + metadata join; ranking is M4.
  CLI: `uv run python -m meme_db.retrieve "facepalm reaction"`.

## Ranking (Milestone 4+)

```
score(meme) =
    0.50 * CLIP_similarity(intent_query, meme)
  + 0.25 * tag_match
  + 0.15 * intensity_match
  + 0.10 * diversity_bonus
  - 0.20 * recent_duplicate_penalty
```

Hand-tune weights first. Learn them from feedback later. Safety penalty is effectively a hard veto.

**Built (M4) in `models/fusion_ranker.py`:** `gesture_to_intent` maps each gesture to a CLIP
query phrase + expected tags + intensity; `FusionRanker` applies the formula above (CLIP
min-max-normalized across the pool, greedy-MMR diversity, a recent-duplicate deque so a
held gesture rotates memes, flagged memes vetoed); `Recommender` wires the OpenCLIP
embedder + vector index + DB. **Order matters:** build the torch embedder *before* loading
faiss (faiss-then-torch segfaults on macOS). Weights are constants now; M7 learns them.
Demo: `scripts/recommend_demo.py` (live webcam) and `scripts/recommend_session.py`
(camera-free, over saved windows). Known weak spot: some CLIP "magnet" memes match many
queries — a magnet penalty is an M7 tuning item.

## Success metric

Per recommendation event: "Did one of the top 5 memes fit the moment?" → **Top-5 Fit Rate**. Optimize this, NOT model accuracy. MVP target 30–40%, decent 50–60%, strong 70%+.

## Never do

- Start end-to-end deep learning, mobile, or a 100k-meme scrape.
- Classify directly into meme IDs.
- Use face expression as ground truth, or let it dominate skeleton.
- Update every frame, skip temporal smoothing, or build UI before skeleton logging works.
- Overclaim emotion.

## Milestone order (do not reorder)

1. Skeleton debugger + sequence logger ✅ done
2. Skeleton features + rule-based gestures ✅ done
3. Meme DB + (subset load →) auto-labeling + CLIP + FAISS ✅ done
4. Skeleton → meme baseline (first demo) ✅ done
5. Face expression module (modifier only) ← **current**
6. Learned temporal motion encoder
7. Fusion ranker + feedback loop
8. Real UI
9. Context-aware mode
10. Personalization loop
