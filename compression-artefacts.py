# ===================================================================

# Example : view jpeg adn png compression artefacts on a video file or live camera
# stream specified on the command line (e.g. python jpeg-compression.py video_file)
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
    description='JPEG and PNG compression artefacts on camera/video image.')

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

window_name = "Live Camera - Compression"  # window name

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

    # settings for the track bars

    jpeg_quality = 20
    cv2.createTrackbar("Compression Quality", window_name, jpeg_quality, 100, lambda x:x)

    amplification = 5
    cv2.createTrackbar("Amplification", window_name, amplification, 255, lambda x:x)

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

        # write/compress and then read back from as JPEG

        quality = cv2.getTrackbarPos("Compression Quality", window_name)
        encode_param_jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        encode_param_png = [int(cv2.IMWRITE_PNG_COMPRESSION), quality//10]

        # either via file output / input
        # only done for jpg here but could be used for png as well

        # cv2.imwrite("camera.jpg", frame, encode_param)
        # jpeg_img = cv2.imread("camera.jpg")

        # or via encoding / decoding in a memory buffer

        retval, buffer = cv2.imencode(".JPG", frame, encode_param_jpeg)        
        jpeg_img = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)

        retval, buffer = cv2.imencode(".PNG", frame, encode_param_png)
        png_img = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)

        # compute absolute difference between original and compressed version

        diff_img_jpg = cv2.absdiff(jpeg_img, frame)
        diff_img_png = cv2.absdiff(png_img, frame)

        # retrieve the amplification setting from the track bar

        amplification = cv2.getTrackbarPos("Amplification", window_name)

        # multiple the result to increase the amplification (so we can see small pixel changes)

        amplified_diff_jpg_img = diff_img_jpg * amplification
        amplified_diff_png_img = diff_img_png * amplification

        # overlay corresponding labels on the images

        cv2.putText(frame, 'Original Input', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(jpeg_img, f'JPEG Compressed - Quality of {quality}', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(amplified_diff_jpg_img, f'| Input - JPEG |', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(png_img, f'PNG Compressed Quality of {quality//10}', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(amplified_diff_png_img, f'| Input - PNG |', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        # pad the input frame so it can be placed in the middle

        frame = cv2.copyMakeBorder(
            frame,
            top=0,
            bottom=0,
            left=math.floor(width/2),
            right=math.ceil(width/2),
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # stack the images into a grid

        im_1 = cv2.hconcat([jpeg_img, amplified_diff_jpg_img])
        im_2 = cv2.hconcat([png_img, amplified_diff_png_img])
        output = cv2.vconcat([frame, im_1, im_2])

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
