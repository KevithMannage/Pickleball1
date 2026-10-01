"""
Re-render annotated frames + an annotated video from an existing prediction
CSV (as produced by infer_video.py), without re-running the model.

Usage:
    python annotate_from_csv.py vidoes/3_3.mp4 predictions/3_3/3_3_pred.csv predictions/3_3 --save-all-frames
"""
import argparse
import os
import os.path as osp

import cv2
import pandas as pd
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("csv")
    ap.add_argument("out_dir")
    ap.add_argument("--save-all-frames", action="store_true",
                     help="save an annotated PNG for every frame, not just detections")
    args = ap.parse_args()

    stem = osp.splitext(osp.basename(args.video))[0]
    frames_dir = osp.join(args.out_dir, f"{stem}_annotated_frames")
    video_out_path = osp.join(args.out_dir, f"{stem}_annotated.mp4")
    os.makedirs(frames_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"cannot open {args.video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {args.video}  {W}x{H} @ {fps:.1f} fps, {n_frames} frames")
    print(f"csv: {args.csv}  {len(df)} rows")

    radius = max(6, int(round(max(W, H) * 0.012)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (W, H))

    n_saved = 0
    i = 0
    rows = df.set_index("Frame").to_dict(orient="index")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row = rows.get(i)
        vis_frame = frame
        visible = bool(row and row.get("Visibility") == 1 and row.get("X") != "" and not pd.isna(row.get("X")))
        if visible:
            x, y = int(round(float(row["X"]))), int(round(float(row["Y"])))
            vis_frame = frame.copy()
            cv2.circle(vis_frame, (x, y), radius, (0, 0, 255), 2)
            cv2.drawMarker(vis_frame, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                            markerSize=radius, thickness=2)
        writer.write(vis_frame)
        if visible or args.save_all_frames:
            cv2.imwrite(osp.join(frames_dir, f"{i}.png"), vis_frame)
            n_saved += 1
        i += 1
    cap.release()
    writer.release()

    print(f"annotated video saved: {video_out_path}")
    print(f"annotated frames saved: {frames_dir} ({n_saved} images)")


if __name__ == "__main__":
    main()
