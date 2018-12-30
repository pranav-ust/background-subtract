# Gamma adjustible background subtractor
# Author : Pranav

import numpy as np
import cv2
from tqdm import *
from skimage.measure import compare_ssim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("video", help = "the video where you want to remove background")
parser.add_argument("back", help = "the background image of the video")
parser.add_argument("--output", help = "the output filename", default = "output.avi")
parser.add_argument("--kernel_size", help = "size of the filtering kernel", type = int, default = 35)
args = parser.parse_args()

# hyperparameters
write_file = args.output
input_file = args.video
background_file = args.back
img_size = args.kernel_size

# load the video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(write_file, fourcc, 10.0, (1920, 1080))
cap = cv2.VideoCapture(input_file)

# captures video length
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# read background file
back_img = cv2.imread(background_file)

# convert background image to black and white
back = cv2.cvtColor(back_img, cv2.COLOR_BGR2GRAY)

# make a kernel of just 1s (averaging filter)
kernel_size = np.ones((img_size, img_size), np.float32) / (img_size * img_size)


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def compare(front, back):
    best_score = -1.0
    optimal_gamma = 0.0
    # compare similarity matrix
    score = compare_ssim(front, back)
    # loop to find best possible gamma
    for gamma in np.arange(0.1, 5.0, 0.1):
        adjusted = adjust_gamma(front, gamma = gamma)
        score = compare_ssim(adjusted, back)
        if score > best_score:
            best_score = score
            optimal_gamma = gamma
        else:
            break

    return adjust_gamma(front, gamma = optimal_gamma), optimal_gamma


for i in tqdm(range(length)):
    # read the frame
    _, frame = cap.read()
    # convert frame to black and white
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # get the optimal gamma of foreground
    foreground, gamma = compare(frame_gray, back.astype(np.uint8))
    # convert into 0-255 matrix
    foreground = foreground.astype(np.uint8)
    background = back.astype(np.uint8)
    # get the difference matrix from foreground and background
    (score, diff) = compare_ssim(foreground, background, full=True)
    # turn difference matrix to 0-255
    diff = (diff * 255).astype("uint8")
    # blur to smoothen edges
    blur = cv2.GaussianBlur(diff, (5,5) ,0)
    # create a mask
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # fill in holes of mask using morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (32, 18))
    morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    final = cv2.filter2D(morph_img, -1, kernel_size)
    # adjust the gamma of the resulting frame
    frame = adjust_gamma(frame, gamma = gamma)
    # do AND operation on resulting mask and adjusted frame
    res = cv2.bitwise_and(frame, frame, mask = final)
    # write the subtracted image to video
    out.write(res)

cap.release()
cv2.destroyAllWindows()
