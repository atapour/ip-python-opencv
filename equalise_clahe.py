# ===================================================================

# Example : clahe equalisation on a video file or live cameran stream 
# from the command line (e.g. python equalise_histogram.py video_file)
# or from an attached web camera by not assigning path to a video file.

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2023 Amir Atapour Abarghouei

# based on : https://github.com/tobybreckon/python-examples-ip/blob/master/skeleton.py
# License : MIT - https://opensource.org/license/mit/

# ===================================================================

import cv2
import argparse
import math
import numpy as np

# ===================================================================

keep_processing = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Perform CLAHE Equalisation on camera/video image.')

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

window_name = "Live Camera - CLAHE Equalisation"  # window name

# if command line arguments are provided try to read video_file
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera))):

    # create window by name (note flags for resizable or not)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # add some track bar controllers for settings

    clip_limit = 2
    cv2.createTrackbar("clip limit", window_name, clip_limit, 25, lambda x:x)
    tile_size = 8
    cv2.createTrackbar("tile size", window_name, tile_size, 64, lambda x:x)

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
        bottomLeftCornerOfText = (10, height - 15)
        fontScale = 1
        fontColor = (123,49,126)
        lineType  = 4

        # rescale image

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # convert to grayscale

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # perform contrast limited adaptive equalization - CLAHE
        # based on example at:
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html

        # get parameters from track bars

        clip_limit = cv2.getTrackbarPos("clip limit", window_name)
        tile_size = cv2.getTrackbarPos("tile size", window_name)

        # perform clahe filtering
        # first create the filter

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        
        # now apply the filter

        output = clahe.apply(gray_img)

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
        cv2.putText(output, 'CLAHE Equalised', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        # stack the images into a grid

        im_1 = cv2.hconcat([gray_img, gray_hist])
        im_2 = cv2.hconcat([output, output_hist])
        output = cv2.vconcat([im_1, im_2])

        # quit instruction label
        
        label = "press 'q' to quit"
        cv2.putText(output, label, (output.shape[1] - 140, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (123,49,126))

        # *******************************

        # stop the timer and convert to milliseconds
        # (to see how long processing and display takes)

        stop_t = ((cv2.getTickCount() - start_t) /
                cv2.getTickFrequency()) * 1000

        label = ('Processing time: %.2f ms' % stop_t) + \
            (' (Max Frames per Second (fps): %.2f' % (1000 / stop_t)) + ')'
        cv2.putText(output, label, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # display image

        cv2.imshow(window_name, output)

        # wait 40ms or less depending on processing time taken (i.e. 1000ms /
        # 25 fps = 40 ms)

        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF

        # It can also be set to detect specific key strokes by recording which
        # key is pressed

        # e.g. if user presses "q" then exit

        if (key == ord('q')):
            keep_processing = False

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

# ===================================================================

# Author : Amir Atapour-Abarghouei
# Copyright (c) 2023 Dept Computer Science, Durham University, UK

# ===================================================================