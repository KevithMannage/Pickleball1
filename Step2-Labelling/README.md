# Step 2: Labelling Tool
[Example Labels File](https://drive.google.com/file/d/198jLZ56IMKi0wlC45YRw_Suvx_nklPa9/view?usp=sharing) \
This is a GUI that helps label the ball location in video footage. It reads the video directly (not the frames from Step 1) and outputs the results as a .csv file with **normalized (0-1) coordinates**, so labels stay valid no matter what resolution the video is at.

## Requirements
```
pip install opencv-python
```

## Step 2.1 — Run the tool
Pass the video (and, optionally, an existing csv to resume) as command-line flags — no need to edit `parser.py`:
```
python labelling_tool.py --label_video_path <videoPath>
```

To continue labelling a video where you left off, also pass the csv you already have:
```
python labelling_tool.py --label_video_path <videoPath> --csv_path <existingLabelsCsv>
```
When a csv is loaded this way, the tool automatically jumps to the first frame **after** the last frame you already clicked a ball position for, instead of starting over at frame 0 — so you can pick up right where you stopped.

(You can still change the defaults inside `parser.py` at lines 44 and 46 if you'd rather not pass flags every time.)

The output csv is written to a dedicated `csv/` folder (created automatically) as `csv/<videoName>.csv`, kept separate from the fixed labels produced in Step 3. To use a different folder, pass `--csv_output_dir <folder>`.

## Controls
| Key | Action |
|-----|--------|
| Left click | Mark ball location at cursor (also sets Ball=1) |
| Middle click | Clear the ball label for this frame (Ball=0) |
| `n` | Next frame |
| `p` | Previous frame |
| `f` | Jump to first frame |
| `l` | Jump to last frame |
| `>` | Fast forward 36 frames |
| `<` | Fast backward 36 frames |
| `s` | Save labels to csv |
| `e` | Exit (warns if unsaved) |

## Output format
```
Frame,Ball,x,y
0,1,0.322,0.624
1,1,0.312,0.630
...
```
- `Frame` — frame index (matches the frame filenames from [Step 1](../Step1-Frames/README.md), e.g. `0.png`, `1.png`, ...)
- `Ball` — 1 if the ball is visible/labelled in this frame, 0 otherwise
- `x`, `y` — ball center, **normalized** to the video's width/height (range 0-1); `-1, -1` when `Ball` is 0

These normalized coordinates are converted to real pixel coordinates on the original frame images in [Step 3](../Step3-FixLabels/README.md).
