# Pickleball Ball Tracking (WASB / HRNet)

Fine-tunes [WASB-SBDT](https://github.com/nttcom/WASB-SBDT) (HRNet-based, originally trained on tennis/badminton/soccer broadcast footage) for **pickleball ball detection**, and ships a full pipeline: label your own match footage → train → run inference on new video, with an online tracker for noise-resistant, gap-filling predictions.

```
vidoes/<video>.mp4
   │
   ▼
Step1-Frames        ──►  frames/<video>_frames/0.png, 1.png, ...
   │
   ▼
Step2-Labelling      ──►  csv/<video>.csv              (normalized 0-1 ball coordinates, hand-labelled)
   │
   ▼
Step3-FixLabels      ──►  fixed_csv/<video>_fixed.csv   (pixel coordinates on the original frame)
   │
   ▼
Step4-Model_Training  ──►  wasb_pickleball_final.pth.zip (fine-tuned checkpoint)
   │
   ▼
infer_video.py        ──►  predictions/<video>/<video>_pred.csv + _annotated.mp4
```

## Model

- **Architecture**: HRNet (verbatim from `nttcom/WASB-SBDT`, Microsoft/MIT-licensed) — full input-resolution heatmap, no downsampling in the stem, so it localizes a ~5px ball where a strided detector can't.
- **I/O**: 3 consecutive RGB frames in (9 input channels), 3 heatmaps out (one per input frame), 512×288 internal resolution via an aspect-preserving affine warp.
- **Starting weights**: the published zero-shot `wasb_tennis_best.pth.tar` checkpoint (outdoor court, similar camera framing/ball scale to pickleball).
- **Fine-tuning**: `Step4-Model_Training/wasb_pickleball_correct_coords.ipynb` (Kaggle notebook, GPU required) trains on your own labelled pickleball clips (`fixed_csv/` + `frames/`) and produces `wasb_pickleball_final.pth.zip`, the checkpoint `infer_video.py` uses by default.

## Quick start — label your own footage

```bash
pip install -r reqirement.txt
```

Put a source video in `vidoes/` (e.g. `vidoes/1.mp4`), then run the three labelling steps in order:

```bash
# Step 1 — extract every frame of the video (native resolution)
python Step1-Frames/video_to_frames.py vidoes/1.mp4 frames/1_frames

# Step 2 — open the GUI, click the ball in each frame (interactive, not scriptable)
python Step2-Labelling/labelling_tool.py --label_video_path vidoes/1.mp4
# writes csv/1.csv  (press 's' to save, 'e' to exit)

# Step 3 — convert normalized labels to pixel coordinates on the original frames
python Step3-FixLabels/fix_labels.py csv/1.csv 1_fixed.csv --frames_dir frames/1_frames
# writes fixed_csv/1_fixed.csv
```

See each step's own README for full details: [Step 1](Step1-Frames/README.md) · [Step 2](Step2-Labelling/README.md) · [Step 3](Step3-FixLabels/README.md).

### Coordinate system, end to end

1. **Step 1** saves frames at the video's native resolution, unmodified.
2. **Step 2** stores click positions **normalized to 0–1** (`x / width`, `y / height`), so the csv is resolution-independent.
3. **Step 3** converts those back to pixel coordinates using the **actual frame image size** from Step 1 (or an explicit `--width`/`--height`) — never a hard-coded size, so `X, Y` always matches the original frame regardless of what resolution the model trains/infers at.

## Training

Once you have labelled clips (`fixed_csv/*.csv` + matching `frames/*_frames/`), open `Step4-Model_Training/wasb_pickleball_correct_coords.ipynb` on a GPU runtime (Kaggle/Colab) — list your clips in the `CLIPS` cell, run through: HRNet build, strict-load the tennis checkpoint, quality-focal-loss fine-tuning, evaluation. It exports `wasb_pickleball_final.pth.zip`; drop that file in the repo root (where `infer_video.py` expects it).

## Inference

```bash
python infer_video.py vidoes/3_3.mp4
```

Uses `wasb_pickleball_final.pth.zip` automatically (falls back to the zero-shot tennis checkpoint, auto-downloaded, if that file isn't present). Outputs land in `predictions/<video_stem>/`:

| file | contents |
|---|---|
| `<stem>_pred.csv` | `Frame, Visibility, X, Y, Score, Source` — pixel coords on the original frame |
| `<stem>_annotated.mp4` | source video with the predicted ball circled in red |
| `<stem>_annotated_frames/` | one PNG per detected frame |

Useful flags:

```bash
python infer_video.py vidoes/3_3.mp4 --weights path/to/other.pth   # use a different checkpoint
python infer_video.py vidoes/3_3.mp4 --tracker median              # disable the online tracker
python infer_video.py vidoes/3_3.mp4 --no-coast                    # strict: invisible when undetected, no extrapolation
python infer_video.py vidoes/3_3.mp4 --search-radius 60            # tighter/looser tracking gate (px)
```

### Online tracker

Per-frame argmax over the heatmap is easily fooled by lines, shoes, glare, or a second ball scoring higher than the real ball. `infer_video.py`'s default `OnlineTracker` (`--tracker online`) instead:

1. fits a short trajectory from recent confirmed points — linear in x, parabolic in y (gravity) — resetting on a bounce (y-direction reversal) or a sudden speed jump (serve/new rally);
2. re-searches the raw heatmap in a radius around the predicted position, at a **lower threshold** than the global detector, so a locally-brightest-but-globally-subthreshold ball can still be recovered;
3. blends the found candidate with the prediction, or coasts on pure extrapolation for a few frames (`--max-age`) if nothing is found nearby — only real detections (not coasted frames) keep the trajectory "warm," so a track that's truly lost a long time resets and re-acquires from scratch rather than drifting forever.

Every row's `Source` column (`global` / `local` / `predicted` / `none`) says which case produced it, so model-detected points stay distinguishable from physics-only guesses.

### Running on a remote GPU server

```bat
infer_remote.bat vidoes\3_3.mp4
```

Edit `REMOTE_USER` / `REMOTE_HOST` / `REMOTE_DIR` at the top of [infer_remote.bat](infer_remote.bat) once. It `scp`s the code + checkpoint + video over, sets up a venv and installs deps on first run, runs `infer_video.py` on the GPU, and `scp`s `predictions/<stem>/` back down. Set up passwordless SSH first (`ssh-copy-id user@host`) so it can run unattended.

## Repository layout

```
vidoes/                   source videos
Step1-Frames/              video → frame images
Step2-Labelling/           GUI labelling tool → normalized csv
Step3-FixLabels/            normalized csv → pixel-coordinate csv
Step4-Model_Training/       fine-tuning notebook → wasb_pickleball_final.pth.zip
frames/                    extracted frames (per-video subfolders)
csv/                       raw labelled csvs (normalized coordinates)
fixed_csv/                 fixed csvs (pixel coordinates)
weights/                   auto-downloaded zero-shot checkpoint (wasb_tennis_best.pth.tar)
wasb_pickleball_final.pth.zip   fine-tuned pickleball checkpoint
infer_video.py             inference: model + online tracker → CSV + annotated video
infer_remote.bat           push code+video to a remote GPU server, run inference, pull results back
annotate_from_csv.py       re-render an annotated video from an existing prediction CSV
convert_pred_to_label_csv.py  prediction CSV → Step 2's label format (for hand-correction)
predictions/               inference output (generated)
```

## Credits

- [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT) — HRNet model, affine-warp preprocessing, heatmap postprocessing (MIT), and the tracking-based postprocessing this project's online tracker is based on. Tarashima et al., *"A Widely Applicable Strong Baseline for Sports Ball Detection and Tracking."*
- HRNet — Microsoft, MIT License.
- Zero-shot starting weights: the published `wasb_tennis_best.pth.tar` checkpoint from the WASB-SBDT model zoo.
