# TrackNetV2 Quick Start

```shell
git clone https://github.com/KevithMannage/Pickleball1.git
cd TrackNetV2
python -m pip install --upgrade pip
python -m pip install --index-url https://pypi.org/simple -r requirements.txt
python detect.py --source ../videos/video.mp4 --weights tracknet_weights.pt --project outputvideo --save-txt
```
