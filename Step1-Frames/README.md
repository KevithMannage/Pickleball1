# Step 1: Turn Video Into Frames
[Example Raw Video](https://drive.google.com/file/d/1_IttM4H7DOy-TL_xemQOnQhvGvwnt7TG/view?usp=sharing) \
In order to label a video, it first needs to be converted from a .mp4 into many .png images (one per frame), at the video's **original resolution**.

## Requirements
```
pip install opencv-python
```

## Step 1.1
```
python video_to_frames.py <videoPath> <framesDirectory>
```
Example:
```
python video_to_frames.py pickleball.mp4 frames/1_1_frames
```

This creates `<framesDirectory>/0.png`, `1.png`, `2.png`, ... (one image per video frame, at the video's native width/height). If `<framesDirectory>` already exists it is deleted and recreated.

The frames produced here are used again in [Step 3](../Step3-FixLabels/README.md) to recover the original image size when fixing labels.
