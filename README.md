# frame-interpolation
Unofficial Pytorch reimplementation of **FILM: Frame Interpolation for Large Motion**.
(Tensorflow 2 implementation can be found here, https://github.com/google-research/frame-interpolation.)


## Installation
Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e. g.
```commandline
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

Install pip dependencies, 
```commandline
pip install -r requirements.txt
```

## Usage

```commandline
INPUT=input.mp4 &&\
OUTPUT=output.mp4 &&\
python run.py -i ${INPUT} -o ${OUTPUT}
```

## Useful ffmpeg commands for video processing

**Cropping.**

```commandline
INPUT=input.mp4 &&\
OUTPUT=output.mp4 &&\
LEFT=10 && TOP=0 && WIDTH=256 && HEIGHT=256
ffmpeg -i ${INPUT} -filter:v "crop=${WIDTH}:${HEIGHT}:${LEFT}:${TOP}" ${OUTPUT}
```

(see, https://ffmpeg.org/ffmpeg-filters.html#toc-crop)

**Changing framerate.**

```commandline
INPUT=input.mp4 &&\
OUTPUT=output.mp4 &&\
FPS=30 &&\
ffmpeg -i ${INPUT} -filter:v fps=${FPS} ${OUTPUT}
```

(see, https://ffmpeg.org/ffmpeg-filters.html#toc-fps-1, or https://trac.ffmpeg.org/wiki/ChangingFrameRate)


**Scaling.**

```commandline
INPUT=input.mp4 &&\
OUTPUT=output.mp4 &&\
W=416 &&\
H=416 &&\
ffmpeg -i ${INPUT} -vf scale=${W}:${H} ${OUTPUT}
```

(see, https://ffmpeg.org/ffmpeg-filters.html#toc-Scaling.)


**Scene into scenes.**

The following command will create a file where for each frame of the specified video a characteristic is calculated, 
how much this frame differs from the previous one.

```commandline
INPUT=input.mp4 &&\
ffmpeg -i ${INPUT} -vf "select='gte(scene,0)',metadata=print:file=scores.txt" -an -f null -
```

**Cut fragment from video.**


```commandline
INPUT=input.mp4 &&\
OUTPUT=output.mp4 &&\
START=00:05:20 &&\
DURATION=00:10:00 &&\
ffmpeg -i ${INPUT} -ss ${START} -t ${DURATION} -c:v copy -c:a copy ${OUTPUT}.
```