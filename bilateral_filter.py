# ===================================================================

# Example : bilateral filter on a video file or live camera stream
# specified on the command line (e.g. python bilateral_filter.py video_file)
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
    description='Perform bilateral filtering on camera/video image with added noise')

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

window_name = "Live Camera Input - Bilateral Filtering"  # window name

# if command line arguments are provided try to read video_file
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera))):

    # create window by name (note flags for resizable or not)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # add some track bar controllers for settings for gaussian filter

    neighbourhood = 10

    cv2.createTrackbar(
        "Gaussian & Mean Neighbourhood",
        window_name,
        neighbourhood,
        40,
        lambda x:x)

    sigma = 50
    cv2.createTrackbar("Gaussian Sigma", window_name, sigma, 200, lambda x:x)

    # add some track bar controllers for settings bilateral filter

    sigma_s = 50
    cv2.createTrackbar("Bilateral Sigma S", window_name, sigma_s, 200, lambda x:x)
    sigma_r = 50
    cv2.createTrackbar("Bilateral Sigma R", window_name, sigma_r, 200, lambda x:x)

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

        # add Gaussian noise to the image

        # get parameters from track bars - gaussian

        neighbourhood = cv2.getTrackbarPos("Gaussian & Mean Neighbourhood", window_name)

        sigma = cv2.getTrackbarPos("Gaussian Sigma", window_name) 

        # get parameters from track bars - bilateral

        sigma_s = cv2.getTrackbarPos("Bilateral Sigma S", window_name)
        sigma_r = cv2.getTrackbarPos("Bilateral Sigma R", window_name)

        # check it is greater than 3 and odd

        neighbourhood = max(3, neighbourhood)
        if not(neighbourhood % 2):
            neighbourhood = neighbourhood + 1

        # perform Gaussian smoothing using NxN neighbourhood

        gaussian_img = cv2.GaussianBlur(
            frame,
            (neighbourhood, neighbourhood),
            sigma,
            sigma,
            borderType=cv2.BORDER_REPLICATE)

         # perform bilateral filtering using a neighbourhood calculated
        # automatically from sigma_s

        bilateral_img = cv2.bilateralFilter(frame, -1, sigma_r, sigma_s, borderType=cv2.BORDER_REPLICATE)

        # Mean filter for comparison

        mean_img = cv2.blur(frame, (neighbourhood,neighbourhood),borderType=cv2.BORDER_DEFAULT)

        # overlay corresponding labels on the images

        cv2.putText(frame, 'Original Input', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(mean_img, f'Mean Filter {neighbourhood}x{neighbourhood}', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(gaussian_img, f'Gaussian Filter {neighbourhood}x{neighbourhood}', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(bilateral_img, f'Bilateral Filter - Sigma {sigma_s} {sigma_r}', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        # stack the images into a grid

        im_1 = cv2.hconcat([frame, mean_img])
        im_2 = cv2.hconcat([gaussian_img, bilateral_img])
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
