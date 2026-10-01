# Step 3 — Fix Labels
The csv from [Step 2](../Step2-Labelling/README.md) stores **normalized** (0-1) ball coordinates. This step converts them into real **pixel coordinates on the original frame image** and fills in any frames the labelling tool skipped (marking them as not visible).

> Previously this script converted coordinates using a hard-coded 512x288 size (the model's training input size), which silently produced wrong pixel coordinates for any video that isn't exactly 512x288. It now auto-detects the true original frame size from the frames folder produced in Step 1 (or the source video), so coordinates always line up with the original images.

## Requirements
```
pip install opencv-python pandas
```

## Step 3.1 — Run the tool
Preferred: point it at the frames folder from [Step 1](../Step1-Frames/README.md) so it can read the real original image size directly:
```
python fix_labels.py <label_csv_path> <new_csv_filename> --frames_dir <framesDirectory>
```
Example:
```
python fix_labels.py ../csv/1_1.csv 1_1_fixed.csv --frames_dir ../frames/1_1_frames
```

Alternatives, if you don't have the frames folder handy:
```
python fix_labels.py <label_csv_path> <new_csv_filename> --video <originalVideoPath>
python fix_labels.py <label_csv_path> <new_csv_filename> --width <W> --height <H>
```

### Output location
If `<new_csv_filename>` is a bare filename (no folder in the path), the fixed csv is saved into a dedicated `fixed_csv/` folder (created automatically), kept separate from the raw Step 2 csvs — e.g. the example above writes `fixed_csv/1_1_fixed.csv`. Pass `--output_dir <folder>` to change that folder, or give `<new_csv_filename>` its own path (e.g. `some/dir/out.csv`) to save exactly there instead.

## Output format
```
Frame,Visibility,X,Y
0,1,618,674
1,1,599,680
...
```
- `Frame` — frame index
- `Visibility` — 1 if the ball was labelled, 0 otherwise
- `X`, `Y` — ball center in **pixel coordinates on the original frame image** (`0, 0` when `Visibility` is 0)

This is the final label format, ready for training.
