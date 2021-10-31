# ===================================================================

# Example : histogram equalisation on a video file or live cameran stream 
# from the command line (e.g. python equalise_histogram.py video_file)
# or from an attached web camera by not assigning path to a video file.

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2021 Amir Atapour Abarghouei

# based on : https://github.com/tobybreckon/python-examples-ip/blob/master/skeleton.py
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# ===================================================================

import cv2
import argparse
import math
import numpy as np

# ===================================================================

keep_processing = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Perform Histogram Equalisation on camera/video image.')

parser.add_argument(
    "--camera",
    type=int,
    help="specify camera to use",
    default=0)

parser.add_argument(
    'video_file',
    metavar='video_file',
    type=str,
    nargs='?',
    help='specify optional video file')

args = parser.parse_args()

# ===================================================================

# basic grayscale histogram drawing in raw OpenCV using lines

# adapted from:
# https://raw.githubusercontent.com/Itseez/opencv/master/samples/python2/hist.py

def hist_lines(hist, width, height):
    h = np.ones((height, width, 3)) * 255  # white background
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist))
    for x, y in enumerate(hist):
        y = y[0]
        cv2.line(h, (x, 0), (x, y), (0, 0, 0))  # black bars
    y = np.flipud(h)
    return y

# ===================================================================

# define video capture object

print("Starting camera stream")
cap = cv2.VideoCapture()

# define display window name

window_name = "Live Camera - Histogram Equalisation"  # window name

# if command line arguments are provided try to read video_file
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera))):

    # create window by name (note flags for resizable or not)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while (keep_processing):

        # if video file or camera successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read()

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False
                continue

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount()

        # *******************************

        # parameters for rescaling the image for easier processing

        scale_percent = 50 # percent of original size
        width = int(frame.shape[1] * scale_percent/100)
        height = int(frame.shape[0] * scale_percent/100)
        dim = (width, height)

        # parameters for overlaying text labels on the displayed images

        font = cv2.FONT_HERSHEY_COMPLEX
        bottomLeftCornerOfText = (10,height-15)
        fontScale = 1
        fontColor = (123,49,126)
        lineType  = 6

        # rescale image

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # convert to grayscale

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # create an empty image of the same size for the output

        output = np.empty(gray_img.shape, dtype=np.uint8)

        # perform histogram equalisation

        output = cv2.equalizeHist(gray_img)

        # convert back to RGB for video stacking

        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        # create the histograms:

        gray_hist = hist_lines(cv2.calcHist([gray_img], [0], None, [256], [0, 256]), 256, height).astype(np.uint8)
        output_hist = hist_lines(cv2.calcHist([output], [0], None, [256], [0, 256]), 256, height).astype(np.uint8)

        # overlay corresponding labels on the images

        cv2.putText(gray_img, 'Original Input', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(output, 'Histogram Equalised', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        # stack the images into a grid

        im_1 = cv2.hconcat([gray_img, gray_hist])
        im_2 = cv2.hconcat([output, output_hist])
        output = cv2.vconcat([im_1, im_2])

        # *******************************

        # display image

        cv2.imshow(window_name, output)
        
        # stop the timer and convert to ms. (to see how long processing and
        # display takes)

        stop_t = ((cv2.getTickCount() - start_t) /
                  cv2.getTickFrequency()) * 1000

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in
        # ms). It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of
        # multi-byte response)

        # wait 40ms or less depending on processing time taken (i.e. 1000ms /
        # 25 fps = 40 ms)

        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

        # It can also be set to detect specific key strokes by recording which
        # key is pressed

        # e.g. if user presses "x" then exit

        if (key == ord('x')):
            keep_processing = False

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

# ===================================================================
