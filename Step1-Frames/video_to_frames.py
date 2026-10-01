import cv2
import os
import sys
import shutil


def video_to_frames(video_path, output_dir):
    """
    Split a video into individual frame images (PNG), one per video frame,
    saved as <output_dir>/0.png, 1.png, 2.png, ...

    param:
    video_path --> path to the source .mp4 video
    output_dir --> folder where extracted frames will be written
                    (created if missing; cleared if it already exists)
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    count = 0
    success, image = cap.read()
    while success:
        frame_path = os.path.join(output_dir, f"{count}.png")
        cv2.imwrite(frame_path, image)
        count += 1
        success, image = cap.read()

    cap.release()
    return count


def main():
    try:
        video_path = sys.argv[1]
        output_dir = sys.argv[2]
        if not video_path or not output_dir:
            raise ValueError
    except (IndexError, ValueError):
        print("usage: python video_to_frames.py <videoPath> <framesDirectory>")
        sys.exit(1)

    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        sys.exit(1)

    count = video_to_frames(video_path, output_dir)
    print(f"Saved {count} frames to '{output_dir}'")


if __name__ == "__main__":
    main()
