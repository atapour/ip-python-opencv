# ===================================================================

# Example : perform bandpass filtering on a video file or live camera stream
# specified on the command line (e.g. python bandpass-filter-fourier.py video_file)
# or from an attached web camera by not assigning path to a video file.

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2021 Amir Atapour Abarghouei

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

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Fourier Transform Band-Pass Filter on camera/video image.')

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

# create a simple band pass filter

def create_band_pass_filter(width, height, radius_small, radius_big):
    assert(radius_big > radius_small)

    bp_filter = np.ones((height, width, 2), np.float32)
    cv2.circle(bp_filter, (int(width / 2), int(height / 2)),
               radius_big, (0, 0, 0), thickness=-1)

    cv2.circle(bp_filter, (int(width / 2), int(height / 2)),
               radius_small, (1, 1, 1), thickness=-1)

    return bp_filter

# ===================================================================

# define video capture object

print("Starting camera stream")
cap = cv2.VideoCapture()

# define display window name

window_name = "Live Camera - Band-Pass Filter"  # window name
window_name2 = "Tracks"
# if command line arguments are provided try to read video_file
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
        or (cap.open(args.camera))):

    # create window by name

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)

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

    # set up optimized DFT settings

    nheight = cv2.getOptimalDFTSize(height)
    nwidth = cv2.getOptimalDFTSize(width)

    # settings for track bars

    radius_big =  100
    radius_small =  75

    cv2.createTrackbar("Big Radius", window_name2, radius_big, 200, lambda x:x)
    cv2.createTrackbar("Small Radius", window_name2, radius_small, 200, lambda x:x)

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

        # convert to grayscale

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Performance of DFT calculation, via the FFT, is better for array
        # sizes of power of two. Arrays whose size is a product of
        # 2's, 3's, and 5's are also processed quite efficiently.
        # Hence we modify the size of the array to the optimal size (by padding
        # zeros) before finding DFT.

        pad_right = nwidth - width
        pad_bottom = nheight - height
        nframe = cv2.copyMakeBorder(
            gray_frame,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0)

        # perform the DFT and get complex output

        dft = cv2.dft(np.float32(nframe), flags=cv2.DFT_COMPLEX_OUTPUT)

        # shift it so that we the zero-frequency, F(0,0), DC component to the
        # center of the spectrum.

        dft_shifted = np.fft.fftshift(dft)

        # get parameters for filter

        radius_small = cv2.getTrackbarPos("Small Radius", window_name2)
        radius_big = cv2.getTrackbarPos("Big Radius", window_name2)

        # do the filtering

        bp_filter = create_band_pass_filter(nwidth, nheight, radius_small, radius_big)
        bp_filter_visual = np.uint8(bp_filter[:,:,0] * 255)


        dft_filtered = cv2.mulSpectrums(dft_shifted, bp_filter, flags=0)

        # shift it back to original quaderant ordering

        dft = np.fft.fftshift(dft_filtered)

        # recover the original image via the inverse DFT

        filtered_img = cv2.dft(dft, flags=cv2.DFT_INVERSE)

        # normalized the filtered image into 0 -> 255 (8-bit grayscale) so we
        # can see the output

        min_val, max_val, min_loc, max_loc = \
            cv2.minMaxLoc(filtered_img[:, :, 0])
        filtered_img_normalised = filtered_img[:, :, 0] * (
            1.0 / (max_val - min_val)) + ((-min_val) / (max_val - min_val))
        filtered_img_normalised = np.uint8(filtered_img_normalised * 255)

        # calculate the magnitude spectrum and log transform + scale it for
        # visualization

        magnitude_spectrum = np.log(cv2.magnitude(
            dft_filtered[:, :, 0], dft_filtered[:, :, 1]))

        # create a 8-bit image to put the magnitude spectrum into

        magnitude_spectrum_normalised = np.zeros((nheight, nwidth, 1), np.uint8)

        # normalized the magnitude spectrum into 0 -> 255 (8-bit grayscale) so
        # we can see the output

        cv2.normalize(
            np.uint8(magnitude_spectrum),
            magnitude_spectrum_normalised,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX)

        # convert back to colour for visualisation

        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        magnitude_spectrum_normalised = cv2.cvtColor(magnitude_spectrum_normalised, cv2.COLOR_GRAY2BGR)
        filtered_img_normalised = cv2.cvtColor(filtered_img_normalised, cv2.COLOR_GRAY2BGR)
        bp_filter_visual = cv2.cvtColor(bp_filter_visual, cv2.COLOR_GRAY2BGR)

        # overlay corresponding labels on the images

        cv2.putText(gray_frame, 'Original Input', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(magnitude_spectrum_normalised, f'Fourier Magnitude Spectrum', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(filtered_img_normalised, f'Filtered Image', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)        
        cv2.putText(bp_filter_visual, f'Mask', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        # stack the images into a grid

        im_1 = cv2.hconcat([gray_frame, bp_filter_visual])
        im_2 = cv2.hconcat([magnitude_spectrum_normalised, filtered_img_normalised])
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

# Amir Atapour-Abarghouei
# Copyright (c) 2023 Dept Computer Science, Durham University, UK

# ===================================================================