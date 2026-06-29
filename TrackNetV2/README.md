# TrackNetV2-pytorch

Paper:	**TrackNetV2: Efﬁcient Shuttlecock Tracking Network**

Original Project（tensorflow）:	https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2

> <del>官方上传的标注工具、数据集均已失效。</del>del>
>
> The author has now reuploaded the dataset。

Paper reading：[TrackNetV2论文记录与pytorch复现](https://zhuanlan.zhihu.com/p/624900770)



## Inference with pytorch weights converted from tensorflow weights:

```shell
git apply tf2torch/diff.txt
python detect.py --source xxx.mp4 --weights  ./tf2torch/track.pt --view-img		# TrackNetv2/3_in_3_out/model906_30
```



## Inference:

```
python detect.py --source xxx.mp4 --weights  xxx.pt --view-img
```

## Measure laptop CPU and RAM usage during detection:

Install dependencies first:

```shell
python -m pip install --upgrade pip
python -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

The default requirements are for inference. If you need ONNX conversion or
deployment helpers, install the optional dependencies separately:

```shell
python -m pip install --index-url https://pypi.org/simple -r requirements-optional.txt
```

Run the detector through the monitor wrapper:

```shell
python monitor_usage.py --csv outputvideo/usage_log.csv --summary outputvideo/usage_summary.json -- python detect.py --source ../videos/video.mp4 --weights tracknet_weights.pt --project outputvideo --save-txt
```

The terminal prints elapsed time, peak RAM, max CPU, and average CPU when the run finishes. The CSV contains one row per sample so you can plot CPU/RAM over time.

## Pickleball court boundary and scoring analysis:

TrackNetV2 predicts the ball position. The added court analysis step uses those predictions to detect the pickleball court boundary, classify ball positions as in/out, estimate bounce events, and maintain a simple team A/team B score timeline.

```shell
python detect.py --source ../../videos/video.mp4 --weights tracknet_weights.pt --save-txt  --project outputvideo
```

Outputs:

```
outputvideo/video_predict.csv              # TrackNet ball positions
outputvideo/video_predict_analysis.csv     # frame-level in/out and score state
outputvideo/video_predict_events.csv       # bounce/out scoring events
outputvideo/video_predict_analysis.mp4     # annotated court, ball, and score video
outputvideo/video_predict_court.json       # detected court model
```

If automatic boundary detection is not accurate for a new camera angle, create or edit a court JSON file and pass it back in:

```json
{
  "boundary": [[650, 210], [1240, 210], [1700, 790], [230, 790]],
  "net_y": 345,
  "near_team": "A",
  "far_team": "B"
}
```

```shell
python detect.py --source ../../videos/video.mp4 --weights tracknet_weights.pt --analyze-court --court-config outputvideo/video_predict_court.json --project outputvideo
```

The current score system is heuristic: it scores out-of-bounds bounce events from ball trajectory only. Full pickleball scoring accuracy requires serve/player-hit context in addition to TrackNet ball centers.



## Training:

```
# training from scratch
python train.py --data data/match.yaml

# training from pretrain weight
python train.py --weights xxx.pt --data data/match.yaml

# resume training
python train.py --data data/match.yaml --resume
```



## Evaluation:

```shell
python val.py --weights xxx.pt --data data/match.yaml
```



## Deployment:

```shell
# Server
python deploy/app.py --weights xxx.pt

# Client
python deploy/test_app.py
```





## Dataset Preparation:

```
# TrackNetV2 dataset
#	/home/chg/Badminton/TrackNetV2
#	- Amateur  
#	- Professional  
#	- Test

python tools/handle_tracknet_dataset.py /home/chg/Badminton/TrackNetV2/Amateur
python tools/handle_tracknet_dataset.py /home/chg/Badminton/TrackNetV2/Professional
python tools/handle_tracknet_dataset.py /home/chg/Badminton/TrackNetV2/Test

python tools/Frame_Generator_rally.py /home/chg/Badminton/TrackNetV2/Amateur
python tools/Frame_Generator_rally.py /home/chg/Badminton/TrackNetV2/Professional
python tools/Frame_Generator_rally.py /home/chg/Badminton/TrackNetV2/Test


# TrackNetV2 dataset config : data/match.yaml
path: /home/chg/Documents/Badminton/TrackNetV2
train:
    - Amateur
    - Professional 
val:
    - Test
    
# also you can use follow config for testing
train:
    - Test/match1/images/1_05_02
val:
    - Test/match2/images/1_03_03

# or
train:
    - Test/match1
val:
    - Test/match2

```



## Reference：

https://github.com/mareksubocz/TrackNet

https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2

https://github.com/ultralytics/yolov5
