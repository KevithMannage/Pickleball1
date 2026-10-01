"""
Convert an infer_video.py prediction CSV (Frame,Visibility,X,Y,Score in pixel
coords) into the format labelling_tool.py expects (Frame,Ball,x,y with x,y
normalized to 0-1, -1,-1 for no-detection frames), so the model's predictions
can be loaded into the GUI and corrected by hand instead of labelled from
scratch.

Usage:
    python convert_pred_to_label_csv.py vidoes/3_3.mp4 D:\pickleball_predictions\3_3\3_3_pred.csv csv/3_3.csv
"""
import argparse
import os.path as osp

import cv2
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("pred_csv")
    ap.add_argument("out_csv")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"cannot open {args.video}"
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    df = pd.read_csv(args.pred_csv)
    rows = df.set_index("Frame").to_dict(orient="index")

    out = []
    for i in range(n_frames):
        row = rows.get(i)
        visible = bool(row and row.get("Visibility") == 1 and not pd.isna(row.get("X")))
        if visible:
            x, y = float(row["X"]) / W, float(row["Y"]) / H
            ball = 1
        else:
            x, y, ball = -1.0, -1.0, 0
        out.append((i, ball, x, y))

    with open(args.out_csv, "w") as f:
        f.write("Frame,Ball,x,y\n")
        for frame, ball, x, y in out:
            f.write(f"{frame},{ball},{x:.6f},{y:.6f}\n")

    n_vis = sum(1 for _, b, _, _ in out if b == 1)
    print(f"wrote {args.out_csv}: {n_frames} frames, {n_vis} pre-filled with a ball position")


if __name__ == "__main__":
    main()
