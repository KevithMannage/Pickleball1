import argparse
import os
import sys
from glob import glob

import cv2
import pandas as pd


def get_original_frame_size(frames_dir=None, video_path=None):
    """
    Determine the width/height of the ORIGINAL frame images (i.e. the frames
    produced by Step 1 / the source video), so normalized label coordinates
    from Step 2 can be converted back into real pixel coordinates on the
    original image instead of some other fixed size.

    param:
    frames_dir  --> folder of frame images from Step 1 (e.g. frames/1_1_frames)
    video_path  --> original source video (alternative to frames_dir)
    """
    if frames_dir:
        candidates = sorted(
            glob(os.path.join(frames_dir, "*.png")) + glob(os.path.join(frames_dir, "*.jpg")),
            key=lambda p: (
                int(os.path.splitext(os.path.basename(p))[0])
                if os.path.splitext(os.path.basename(p))[0].isdigit()
                else os.path.basename(p)
            ),
        )
        if not candidates:
            raise FileNotFoundError(f"No frame images (.png/.jpg) found in '{frames_dir}'")

        img = cv2.imread(candidates[0])
        if img is None:
            raise ValueError(f"Could not read image '{candidates[0]}'")
        h, w = img.shape[:2]
        return w, h

    if video_path:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w == 0 or h == 0:
            raise ValueError(f"Could not read frame size from video '{video_path}'")
        return w, h

    raise ValueError(
        "Must provide --frames_dir, --video, or both --width and --height "
        "so the original image size is known."
    )


def fix_labels(input_csv, frame_width, frame_height):
    """
    Convert the normalized (0-1) x/y coordinates written by the Step 2
    labelling tool into pixel coordinates on the ORIGINAL frame image, and
    fill in any frames that were skipped (no ball visible) with Visibility=0.

    param:
    input_csv     --> label csv produced by Step 2 (columns: Frame,Ball,x,y)
    frame_width   --> width in pixels of the original frame images
    frame_height  --> height in pixels of the original frame images
    """
    raw = pd.read_csv(input_csv)

    frames, visibility, xs, ys = [], [], [], []
    for _, row in raw.iterrows():
        frame = int(row["Frame"])
        ball = int(row["Ball"])
        frames.append(frame)
        visibility.append(ball)
        if ball == 1:
            xs.append(round(float(row["x"]) * frame_width))
            ys.append(round(float(row["y"]) * frame_height))
        else:
            xs.append(0)
            ys.append(0)

    df_label = pd.DataFrame({"Frame": frames, "Visibility": visibility, "X": xs, "Y": ys})

    # Compensate for any frame indices missing from the input csv
    existing = set(df_label["Frame"])
    last_frame = max(frames) if frames else -1
    missing_rows = [
        {"Frame": i, "Visibility": 0, "X": 0, "Y": 0}
        for i in range(0, last_frame + 1)
        if i not in existing
    ]
    if missing_rows:
        df_label = pd.concat([df_label, pd.DataFrame(missing_rows)], ignore_index=True)

    df_label = df_label.sort_values(by=["Frame"]).reset_index(drop=True)
    return df_label


def main():
    ap = argparse.ArgumentParser(
        description="Convert normalized ball-label coordinates from Step 2 "
        "into pixel coordinates on the ORIGINAL frame image."
    )
    ap.add_argument("input_csv", help="label csv produced by the labelling tool (Step 2)")
    ap.add_argument(
        "output_csv",
        help="filename (or path) to write the fixed csv to; if just a filename with no "
        "directory is given, it is saved under --output_dir so fixed labels stay separate "
        "from the raw Step 2 csvs",
    )
    ap.add_argument(
        "--output_dir",
        default="fixed_csv",
        help="folder to save the fixed csv into when output_csv has no directory of its own "
        "(default: fixed_csv)",
    )
    ap.add_argument(
        "--frames_dir",
        default=None,
        help="folder of frames from Step 1 (used to auto-detect the original image size), "
        "e.g. frames/1_1_frames",
    )
    ap.add_argument(
        "--video",
        default=None,
        help="original source video (alternative to --frames_dir for detecting image size)",
    )
    ap.add_argument("--width", type=int, default=None, help="override: original frame width in pixels")
    ap.add_argument("--height", type=int, default=None, help="override: original frame height in pixels")
    args = ap.parse_args()

    if not os.path.isfile(args.input_csv):
        print(f"Input csv not found: {args.input_csv}")
        sys.exit(1)

    if args.width and args.height:
        frame_width, frame_height = args.width, args.height
    else:
        try:
            frame_width, frame_height = get_original_frame_size(args.frames_dir, args.video)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            print(
                "usage: python fix_labels.py <label_csv_path> <new_csv_path> "
                "(--frames_dir <framesFolder> | --video <videoPath> | --width W --height H)"
            )
            sys.exit(1)

    print(f"Using original image size: {frame_width}x{frame_height}")

    output_csv = args.output_csv
    out_dirname = os.path.dirname(output_csv)
    if not out_dirname and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_csv = os.path.join(args.output_dir, output_csv)
    elif out_dirname:
        os.makedirs(out_dirname, exist_ok=True)

    df_label = fix_labels(args.input_csv, frame_width, frame_height)
    df_label.to_csv(output_csv, encoding="utf-8", index=False)
    print(f"Saved fixed labels to '{output_csv}'")


if __name__ == "__main__":
    main()
