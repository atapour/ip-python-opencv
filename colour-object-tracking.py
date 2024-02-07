# ===================================================================

# Example: perform colour object tracking on a video file or live
# camera stream specified on the command line
# (e.g. python colour-object-tracking.py video_file)
# or from an attached web camera by not assigning path to a video.
# This script takes advantage of the mean-shift algorithm.

# Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2024 Amir Atapour Abarghouei

# based on : https://github.com/tobybreckon/python-examples-ip/blob/master/skeleton.py
# and : https://docs.opencv.org/3.4/d7/d00/tutorial_meanshift.html

# License : MIT - https://opensource.org/license/mit/

# ===================================================================

import cv2
import argparse
import math
import numpy as np
import warnings

# ===================================================================

warnings.filterwarnings("ignore")
keep_processing = True
selection_in_progress = False  # support interactive region selection

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Colour object tracking on camera/video image.')

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

# select a region using the mouse

boxes = []
current_mouse_position = np.ones(2, dtype=np.int32)


def on_mouse(event, x, y, flags, params):

    global boxes
    global selection_in_progress

    current_mouse_position[0] = x
    current_mouse_position[1] = y

    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = []
        sbox = [x, y]
        selection_in_progress = True
        boxes.append(sbox)

    elif event == cv2.EVENT_LBUTTONUP:
        ebox = [x, y]
        selection_in_progress = False
        boxes.append(ebox)

# ===================================================================

# define video capture object

print("Starting camera stream")
cap = cv2.VideoCapture()

# define display window name

window_name = "Live Camera - Colour Object Tracking"  # window name
window_name2 = "Track Bars"  # window name


# if command line arguments are provided try to read video_file
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera))):

    # create window by name

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)

    # add track bar controllers for settings for HSV selection thresholds

    s_lower = 60
    cv2.createTrackbar("S lower", window_name2, s_lower, 255, lambda x:x)
    s_upper = 255
    cv2.createTrackbar("S upper", window_name2, s_upper, 255, lambda x:x)
    v_lower = 32
    cv2.createTrackbar("V lower", window_name2, v_lower, 255, lambda x:x)
    v_upper = 255
    cv2.createTrackbar("V upper", window_name2, v_upper, 255, lambda x:x)

    # capture one frame just for settings

    if (cap.isOpened):
            ret, frame = cap.read()

    # parameters for rescaling the image for easier processing

    scale_percent = 100 # percent of original size
    width = int(frame.shape[1] * scale_percent/100)
    height = int(frame.shape[0] * scale_percent/100)
    dim = (width, height)

    # set a mouse callback

    cv2.setMouseCallback(window_name, on_mouse, 0)
    cropped = False

    # set up the termination criteria for search, either 10 iteration or
    # move by at least 1 pixel pos. difference
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # usage

    print("USAGE: click and drag left to right to select an image region")

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

        # parameters for overlaying text labels on the displayed images

        font = cv2.FONT_HERSHEY_COMPLEX
        bottomLeftCornerOfText = (10,height-15)
        fontScale = 1
        fontColor = (123,49,126)
        lineType  = 6

        # rescale image

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # get parameters from track bars

        s_lower = cv2.getTrackbarPos("S lower", window_name2)
        s_upper = cv2.getTrackbarPos("S upper", window_name2)
        v_lower = cv2.getTrackbarPos("V lower", window_name2)
        v_upper = cv2.getTrackbarPos("V upper", window_name2)

        # select region using the mouse and display it

        if (len(boxes) > 1) and (boxes[0][1] < boxes[1][1]) and (
                boxes[0][0] < boxes[1][0]):
            crop = frame[boxes[0][1]:boxes[1][1],
                         boxes[0][0]:boxes[1][0]].copy()

            h, w, c = crop.shape   # size of template

            if (h > 0) and (w > 0):
                cropped = True

                # convert region to HSV

                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

                # select all Hue (0-> 180) and Sat. values but eliminate values
                # with very low saturation or value (due to lack of useful
                # colour information)

                mask = cv2.inRange(
                    hsv_crop, np.array(
                        (0., float(s_lower), float(v_lower))), np.array(
                        (180., float(s_upper), float(v_upper))))

                # construct a histogram of hue and saturation values and
                # normalize it

                crop_hist = cv2.calcHist(
                    [hsv_crop], [
                        0, 1], mask, [
                        180, 255], [
                        0, 180, 0, 255])
                cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)

                # set initial position of object

                track_window = (
                    boxes[0][0],
                    boxes[0][1],
                    boxes[1][0] -
                    boxes[0][0],
                    boxes[1][1] -
                    boxes[0][1])


            # reset list of boxes

            boxes = []

        # interactive display of selection box

        if (selection_in_progress):
            top_left = (boxes[0][0], boxes[0][1])
            bottom_right = (
                current_mouse_position[0],
                current_mouse_position[1])
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # if we have a selected region

        if cropped:

            # convert incoming image to HSV

            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            img_bproject = cv2.calcBackProject(
                [img_hsv], [
                    0, 1], crop_hist, [
                    0, 180, 0, 255], 1)

            # apply meanshift to get the new location

            ret, track_window = cv2.meanShift(img_bproject, track_window, term_crit)

            # Draw it on image
            x, y, w, h = track_window

            frame = cv2.rectangle(
                frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            img_bproject = cv2.cvtColor(img_bproject, cv2.COLOR_GRAY2BGR)

            # getting the size of the crop so it can be overlaid

            crop_width = int(crop.shape[1])
            crop_height = int(crop.shape[0])

            # overlay text on images

            cv2.putText(img_bproject, 'Meanshift - Hue', 
                (10, img_bproject.shape[0]-15), 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(crop, 'Selection', 
                (10, crop.shape[0]-15),
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(frame, 'Input Image', 
                (10, frame.shape[0]-15), 
                font, 
                fontScale,
                fontColor,
                lineType)

            # overlay the crop on top of the mask

            img_bproject[0:crop_height,0:crop_width,:] = crop

            # concat images to eachother

            output = cv2.hconcat([frame, img_bproject])

        else:

            # before we have cropped anything show the mask we are using
            # for the S and V components of the HSV image

            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # select all Hue values (0-> 180) but eliminate values with very
            # low saturation or value (due to lack of useful colour info.)

            mask = cv2.inRange(
                img_hsv, np.array(
                    (0., float(s_lower), float(v_lower))), np.array(
                    (180., float(s_upper), float(v_upper))))

            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            cv2.putText(mask, 'Mask for S and V', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

            cv2.putText(frame, 'USAGE: click and drag left to right', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

            output = cv2.hconcat([frame, mask])

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
# Copyright (c) 2024 Dept Computer Science, Durham University, UK

# ===================================================================