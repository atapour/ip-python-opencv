# ===================================================================

# Example : nonlocal means filter on a video file or live camera stream
# specified on the command line (e.g. python nlm_filter.py video_file)
# or from an attached web camera by not assigning path to a video file.

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2024 Amir Atapour Abarghouei

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
    description='Perform non-local means filtering on camera/video image with added noise')

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

window_name = "Live Camera - Non-Local Means"  # window name

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

    # add some track bar controllers for settings nlm filter

    search_window = 21
    cv2.createTrackbar("NLM search Area", window_name,
                       search_window, 50, lambda x:x)
    filter_strength = 10
    cv2.createTrackbar(
        "NLM Strength",
        window_name,
        filter_strength,
        25,
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
        lineType  = 4

        # rescale image

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # add salt and Pepper noise:

        s_vs_p = 0.5
        amount = 0.02
        noisy_sp = np.copy(frame)

        # salt mode

        num_salt = np.ceil(amount * frame.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in frame.shape]
        noisy_sp[tuple(coords)] = 255

        # pepper mode

        num_pepper = np.ceil(amount* frame.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in frame.shape]
        noisy_sp[tuple(coords)] = 0

        # get parameters from track bars - gaussian

        neighbourhood = cv2.getTrackbarPos("Gaussian & Mean Neighbourhood", window_name)

        sigma = cv2.getTrackbarPos("Gaussian Sigma", window_name) 

        # get parameters from track bars - nlm

        search_window = cv2.getTrackbarPos("NLM search Area", window_name)
        filter_strength = cv2.getTrackbarPos("NLM Strength", window_name)

        # check it is greater than 3 and odd

        neighbourhood = max(3, neighbourhood)
        if not(neighbourhood % 2):
            neighbourhood = neighbourhood + 1

        # perform Gaussian smoothing using NxN neighbourhood

        gaussian_img = cv2.GaussianBlur(
            noisy_sp,
            (neighbourhood, neighbourhood),
            sigma,
            sigma,
            borderType=cv2.BORDER_REPLICATE)

         # perform bilateral filtering using a neighbourhood calculated
        # automatically from sigma_s

        nlm_img = cv2.fastNlMeansDenoisingColored(
            noisy_sp,
            h=filter_strength,
            hColor=10,
            templateWindowSize=neighbourhood,
            searchWindowSize=search_window)

        # Mean filter for comparison

        mean_img = cv2.blur(noisy_sp, (neighbourhood,neighbourhood),borderType=cv2.BORDER_DEFAULT)

        # overlay corresponding labels on the images

        cv2.putText(noisy_sp, 'Salt n Pepper Input', 
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
        cv2.putText(gaussian_img, f'Gaussian Filter {neighbourhood}x{neighbourhood} - {sigma}', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(nlm_img, f'NLM Filter', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        # stack the images into a grid

        im_1 = cv2.hconcat([noisy_sp, mean_img])
        im_2 = cv2.hconcat([gaussian_img, nlm_img])
        output = cv2.vconcat([im_1, im_2])

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

# Author : Amir Atapour-Abarghouei
# Copyright (c) 2024 Dept Computer Science, Durham University, UK

# ===================================================================
