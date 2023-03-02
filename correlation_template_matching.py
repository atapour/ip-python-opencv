# ===================================================================

# Example : perform template matching via correlation on a video file or live camera stream
# specified on the command line (e.g. python correlation_template_matching.py video_file)
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
import warnings

# ===================================================================

warnings.filterwarnings("ignore")
keep_processing = True
selection_in_progress = False  # support interactive region selection

# ===================================================================

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Correlation Template Matching on camera/video image.')

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

window_name = "Live Camera - Template Matching"

# if command line arguments are provided try to read video_file
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera))):

    # create window by name

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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
        lineType  = 4

        # rescale image

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # select region using the mouse and display it

        if (len(boxes) > 1) and (boxes[0][1] < boxes[1][1]) and (boxes[0][0] < boxes[1][0]):
            crop = frame[boxes[0][1]:boxes[1][1], boxes[0][0]:boxes[1][0]].copy()
            boxes = []
            h, w, c = crop.shape   # size of template
            if (h > 0) and (w > 0):
                cropped = True

        # interactive display of selection box

        if (selection_in_progress):
            top_left = (boxes[0][0], boxes[0][1])
            bottom_right = (current_mouse_position[0], current_mouse_position[1])
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # if we have cropped a region perform template matching using
        # (normalized) cross correlation and draw rectangle around best match

        if cropped:
            correlation = cv2.matchTemplate(frame, crop, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation)
            h, w, c = crop.shape   # size of template
            top_left = max_loc     # top left of template matching image frame
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

            # visualising correlation

            corr = np.uint8(correlation * 255) 

            # sticking the crop and correlation images to the frame

            crop_width = int(crop.shape[1])
            crop_height = int(crop.shape[0])

            # resize correlation image

            resize_corr = cv2.resize(corr, (crop_width, height - crop_height), interpolation = cv2.INTER_AREA)
            resize_corr = cv2.cvtColor(resize_corr, cv2.COLOR_GRAY2BGR)

            # overlay text on images

            cv2.putText(resize_corr, 'Corr', 
                (10, resize_corr.shape[0]-15), 
                font, 
                fontScale,
                fontColor,
                lineType)
            cv2.putText(crop, 'Crop', 
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

            # concat images to eachother

            im_1 = cv2.vconcat([resize_corr, crop])
            output = cv2.hconcat([frame, im_1])

        else:

            # cropping has not happened yet - so only displaying
            # the input frame for the user to crop

            output = frame.copy()
            cv2.putText(output, 'USAGE: click and drag left to right to select an image region', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

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
