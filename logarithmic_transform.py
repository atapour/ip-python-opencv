# ===================================================================

# Example : logarithmic transform on a video file or live camera stream
# specified on the command line (e.g. python logarithmic_transform.py video_file)
# or from an attached web camera

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2024 Amir Atapour Abarghouei

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
    description='Perform logarithmic transform on camera/video image')

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


# logarithmic transform
# image - greyscale image

def logarithmic_transform(image):

    image = image / 2
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    log_image = np.array(log_image, dtype = np.uint8)

    return log_image
# ===================================================================

# define video capture object

print("Starting camera stream")
cap = cv2.VideoCapture()

# define display window name

window_name = "Live Camera Input and Logarithmic Transform"  # window name

# if command line arguments are provided try to read video_file
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera))):

    # create window by name

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

        scale_percent = 70 # percent of original size
        width = int(frame.shape[1] * scale_percent/100)
        height = int(frame.shape[0] * scale_percent/100)
        dim = (width, height)

        # rescale image

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # convert to grayscale

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # make a copy and exp transform it

        log_img = logarithmic_transform(gray_img)

        # parameters for overlaying text labels on the displayed images

        font = cv2.FONT_HERSHEY_COMPLEX
        bottomLeftCornerOfText = (10,height-15)
        fontScale = 1
        fontColor = (123,49,126)
        lineType  = 6

        # convert to 3 channel for colour labels

        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        log_img = cv2.cvtColor(log_img, cv2.COLOR_GRAY2BGR)

        # overlay corresponding labels on the images

        cv2.putText(gray_img, 'Original Grayscale', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(log_img, f'Logarithmic Transform', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        # stack the images into a grid

        output = cv2.hconcat([gray_img, log_img])

        # quit instruction label
        
        label = "press 'q' to quit"
        cv2.putText(output, label, (output.shape[1] - 150, 20),
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

# Amir Atapour-Abarghouei
# Copyright (c) 2024 Dept Computer Science, Durham University, UK

# ===================================================================
