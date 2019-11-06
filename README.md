#### Prerequisitives:


Install all the requirements using python3.6+
```
virtualenv venv -p python3.6
. venv/bin/activate
pip install -r requirements.txt
```

#### Usage:

```
Usage: my_tool.py [-h] [-v VIDEO_PATH] [-m MODEL] [-k TOP_K]

Process video and collect most common classes

optional arguments:
  -h, --help            show this help message and exit
  -v VIDEO_PATH, --video_path VIDEO_PATH
                        Path to the video file
  -m MODEL, --model MODEL
                        Pretrained ImageNet model name from torchvision module
  -k TOP_K, --top_k TOP_K
                        How many most common classes to return
```

#### Example:
```
python my_tool.py -v video.mp4 -k 3 -m wide_resnet50_2
```
Result:
```
Most common classes: {'bighorn': 51, 'geyser': 31, 'Arctic_fox': 29}
```



#### Possible improvements:
- add gpu support if available
- use normal logging instead of prints
- possibly process frames in parallel