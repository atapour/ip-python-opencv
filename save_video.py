# ===================================================================

# Example : save video from live camera stream rom an attached
# web camera by not assigning path to a video file.

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2024 Amir Atapour Abarghouei

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
    description='Save video file to disk from camera stream')

parser.add_argument(
    "--camera",
    type=int,
    help="specify camera to use",
    default=0)

args = parser.parse_args()

# ===================================================================

# define video size

video_width = 640
video_height = 480

# define video capture object

print("Starting camera stream")
cap = cv2.VideoCapture()

# define display window name

window_name = "Live Camera -> Video File"  # window name

# define video writer (video: 640 x 480 @ 25 fps)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
output = cv2.VideoWriter('output.avi', fourcc, 25.0, (video_width, video_height))

# capture from attached H/W camera

if ((cap.open(args.camera))):

    # create window by name (note flags for resizable or not)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # add some track bar controllers for settings for gaussian filter

    while (keep_processing):

        # if video file or camera successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read()

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False
                continue

        # start a timer (to see how long processing and display takes if needed)

        start_t = cv2.getTickCount()

        # *******************************

        # do any processing here

        # resize the video to the size defined at the top of the program

        frame2 = cv2.resize(
            frame,
            (video_width,
             video_height),
            interpolation=cv2.INTER_CUBIC)
        output.write(frame2)

        # *******************************

        # display image

        cv2.imshow(window_name, frame)
        
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

    # Release everything when the job is finished

    cap.release()
    output.release()

else:
    print("No video file specified or camera connected.")

# ===================================================================

# Author : Amir Atapour-Abarghouei
# Copyright (c) 2024 Dept Computer Science, Durham University, UK

# ===================================================================
