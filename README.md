# Video Background Subtraction with Adaptive Gamma Correction

I made a handy background subtraction tool which is generally useful as a preprocessing for object detection.

The situations I dealt with often had extreme changes in light. So this background subtraction tool takes brightness changes into account, normalizes it to produce a more stable video.

![Demo](https://github.com/pranav-ust/transfer/blob/master/animation.gif)

## Requirements

You need:

1. Python3
2. TQDM
3. OpenCV 3+
4. Scikit Image
5. Input video from where the background has to be removed.
6. Background image of that video

## Usage

Run the file `python3 subtract.py video background`

More details are as follows:

```
usage: subtract.py [-h] [--output OUTPUT] [--kernel_size KERNEL_SIZE]
                   video background

positional arguments:
  video                 the video that you want to input
  background		background image of that video

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       the output filename
  --kernel_size KERNEL_SIZE
                        size of the filtering kernel
```

