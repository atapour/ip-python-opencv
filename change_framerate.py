# ===================================================================

# Example : change the framerate on a video file or live camera stream
# specified on the command line (e.g. python change_framerate.py video_file)
# or from an attached web camera by not assigning path to a video file.

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2022 Amir Atapour Abarghouei

# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# ===================================================================

import cv2
import argparse
import math

# ===================================================================

keep_processing = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Change the framerate of camera/video sequence.')

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

# define video capture object

print("Starting camera stream")
cap = cv2.VideoCapture()




# define display window name

window_name = "Live Camera Input - Framerate"  # window name

# if command line arguments are provided try to read video_file
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera))):

    cam_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # create window by name (note flags for resizable or not)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # add some track bar controllers for settings

    framerate = cam_fps - 20
    cv2.createTrackbar(
        "FPS",
        window_name,
        framerate,
        cam_fps,
        lambda x:x)

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

        # get parameter from track bar

        framerate = cv2.getTrackbarPos("FPS", window_name)

        # parameters for overlaying text labels on the displayed images

        font = cv2.FONT_HERSHEY_COMPLEX
        bottomLeftCornerOfText = (10, int(frame.shape[0])-15)
        fontScale = 1
        fontColor = (123,49,126)
        lineType  = 6

        # overlay corresponding labels on the images

        cv2.putText(frame, f'Camera Framerate: {cam_fps} - Current Framerate: {framerate} fps', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        # *******************************

        # display image

        cv2.imshow(window_name, frame)
        
        # framerate cannot be zero:

        if framerate == 0:
            framerate = 1

        # using waitKey to simulate framerate

        key = cv2.waitKey(int(1000/framerate))

        # The loop can be set to detect specific key strokes by recording which
        # key is pressed

        # e.g. if user presses "x" then exit

        if (key == ord('x')):
            keep_processing = False

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

# ===================================================================
