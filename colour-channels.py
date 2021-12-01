# ===================================================================

# Example: display channels of different colour spaces on a 
# video file or live  camera stream specified on the command line 
# (e.g. python colour-channels.py video_file)
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
import warnings

# ===================================================================

warnings.filterwarnings("ignore")
keep_processing = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Display colour channels from a camera/video image.')

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
colour_map =  False

# define display window name

window_name = "Live Camera - Colour Channels"  # window name

print("USAGE: press 'c' to toggle some colour mapping!")

# if command line arguments are provided try to read video_file
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera))):

    # create window by name

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # capture one frame just for settings

    if (cap.isOpened):
            ret, frame = cap.read()

    # convert to grayscale

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # parameters for rescaling the image for easier processing

    scale_percent = 50 # percent of original size
    width = int(gray_frame.shape[1] * scale_percent/100)
    height = int(gray_frame.shape[0] * scale_percent/100)
    dim = (width, height)

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

        rgb = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # get RGB channels separately

        red, green, blue = cv2.split(rgb)

        # convert image to hsv

        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        # get HSV channels separately

        saturation = hsv[:, :, 1].copy()
        value = hsv[:, :, 2].copy()

        if (colour_map):
                # re map S and V to top outer rim of HSV colour space

            hsv[:, :, 1] = np.ones(hsv[:, :, 1].shape) * 255
            hsv[:, :, 2] = np.ones(hsv[:, :, 1].shape) * 255

            # convert the result back to BGR to produce a false colour
            # version of hue for display

            hue = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            hue = hsv[:, :, 0]

        # convert images to lab

        lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)

        # get LAB channels separately

        # luminance, a, b = cv2.split(lab)
        # get HSV channels separately

        a = lab[:, :, 1].copy()
        b = lab[:, :, 2].copy()

        if (colour_map):
                # re map S and V to top outer rim of HSV colour space

            lab[:, :, 1] = np.ones(lab[:, :, 1].shape) * 255
            lab[:, :, 2] = np.ones(lab[:, :, 1].shape) * 255

            # convert the result back to BGR to produce a false colour
            # version of hue for display

            luminance = cv2.cvtColor(lab, cv2.COLOR_HSV2BGR)
        else:
            luminance = lab[:, :, 0]

        # convert back to colour for visualisation

        red = cv2.cvtColor(red, cv2.COLOR_GRAY2BGR)
        green = cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)
        blue = cv2.cvtColor(blue, cv2.COLOR_GRAY2BGR)
        value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
        saturation = cv2.cvtColor(saturation, cv2.COLOR_GRAY2BGR)
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

        if (colour_map == False):
            hue = cv2.cvtColor(hue, cv2.COLOR_GRAY2BGR)
            luminance = cv2.cvtColor(luminance, cv2.COLOR_GRAY2BGR)
        else:             
            red[:,:,0] = 0
            red[:,:,1] = 0
            green[:,:,0] = 0
            green[:,:,2] = 0
            blue[:,:,2] = 0
            blue[:,:,1] = 0
        # overlay corresponding labels on the images

        cv2.putText(rgb, 'Input', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(red, 'R', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(green, f'G', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(blue, f'B', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(hue, f'H', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)        
        cv2.putText(saturation, f'S', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)   
        cv2.putText(value, f'V', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType) 
        cv2.putText(luminance, f'L', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType) 
        cv2.putText(a, f'a', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType) 
        cv2.putText(b, f'b', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType) 


        # stack the images into a grid

        im_1 = cv2.hconcat([rgb, red, green, blue])
        im_2 = cv2.hconcat([rgb, hue, saturation, value])
        im_3 = cv2.hconcat([rgb, luminance, a, b])
        output = cv2.vconcat([im_1, im_2, im_3])

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
        elif (key == ord('c')):
            colour_map = not(colour_map)
    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")

# ===================================================================
