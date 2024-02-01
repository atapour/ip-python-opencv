# ===================================================================

# Example : perform high and low pass butterworth filtering on a video 
# file or live camera stream specified on the command line 
# (e.g. python butterworth-low-high-pass-filter.py video_file)
# or from an attached web camera by not assigning path to a video file.

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2024 Amir Atapour Abarghouei

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
recompute_filter = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Fourier Transform High/Low-Pass Butterworth Filter on camera/video image.')

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

# create a butterworth high pass filter

def create_butterworth_high_pass_filter(width, height, d, n):
    hp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width / 2, height / 2)

    for i in range(0, hp_filter.shape[1]):  # image width
        for j in range(0, hp_filter.shape[0]):  # image height
            radius = max(1, math.sqrt(math.pow((i - centre[0]), 2.0) + math.pow((j - centre[1]), 2.0)))
            hp_filter[j, i] = 1 / (1 + math.pow((d / radius), (2 * n)))
    return hp_filter


# create a butterworth low pass filter

def create_butterworth_low_pass_filter(width, height, d, n):
    lp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width / 2, height / 2)

    for i in range(0, lp_filter.shape[1]):  # image width
        for j in range(0, lp_filter.shape[0]):  # image height
            radius = max(1, math.sqrt(math.pow((i - centre[0]), 2.0) + math.pow((j - centre[1]), 2.0)))
            lp_filter[j, i] = 1 / (1 + math.pow((radius / d), (2 * n)))
    return lp_filter


# this function is called everytime the trackbar is moved to signal we need to reconstruct the filter

def reset_butterworth_filter(_):
    global recompute_filter
    recompute_filter = True
    return

# ===================================================================

# define video capture object

print("Starting camera stream")
cap = cv2.VideoCapture()

# define display window name

window_name = "Live Camera - High/Low-Pass Butterworth Filter"  # window name

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

    # set up optimized DFT settings

    nheight = cv2.getOptimalDFTSize(height)
    nwidth = cv2.getOptimalDFTSize(width)

    # settings for the track bars

    cv2.createTrackbar("Radius", window_name, 5, 200, reset_butterworth_filter)
    cv2.createTrackbar("Order", window_name, 1, 20, reset_butterworth_filter)

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

        # get parameters for filters

        radius = cv2.getTrackbarPos("Radius", window_name)
        order = cv2.getTrackbarPos("Order", window_name)

        # do the filtering

        # butterworth is slow to construct so we'll only do it when needed (i.e. trackbar changes)

        if (recompute_filter):
            lp_filter = create_butterworth_low_pass_filter(nwidth, nheight, radius, order)
            hp_filter = create_butterworth_high_pass_filter(nwidth, nheight, radius, order)
            recompute_filter = False

        hi_dft_filtered = cv2.mulSpectrums(dft_shifted, hp_filter, flags=0)
        lo_dft_filtered = cv2.mulSpectrums(dft_shifted, lp_filter, flags=0)

        # shift back to original quaderant ordering

        hi_dft = np.fft.fftshift(hi_dft_filtered)
        lo_dft = np.fft.fftshift(lo_dft_filtered)

        # recover the original image via the inverse DFT

        hi_filtered_img = cv2.dft(hi_dft, flags=cv2.DFT_INVERSE)
        lo_filtered_img = cv2.dft(lo_dft, flags=cv2.DFT_INVERSE)

        # normalized the filtered image into 0 -> 255 (8-bit grayscale) 
        # so we can see the output

        # high pass filter output

        hi_min_val, hi_max_val, hi_min_loc, hi_max_loc = \
            cv2.minMaxLoc(hi_filtered_img[:, :, 0])
        hi_filtered_img_normalised = hi_filtered_img[:, :, 0] * (
            1.0 / (hi_max_val - hi_min_val)) + ((-hi_min_val) / (hi_max_val - hi_min_val))
        hi_filtered_img_normalised = np.uint8(hi_filtered_img_normalised * 255)

        # low pass filter output

        lo_min_val, lo_max_val, lo_min_loc, lo_max_loc = \
            cv2.minMaxLoc(lo_filtered_img[:, :, 0])
        lo_filtered_img_normalised = lo_filtered_img[:, :, 0] * (
            1.0 / (lo_max_val - lo_min_val)) + ((-lo_min_val) / (lo_max_val - lo_min_val))
        lo_filtered_img_normalised = np.uint8(lo_filtered_img_normalised * 255)

        # calculate the magnitude spectrum and log transform + scale for visualization

        hi_magnitude_spectrum = np.log(cv2.magnitude(
            hi_dft_filtered[:, :, 0], hi_dft_filtered[:, :, 1]))

        lo_magnitude_spectrum = np.log(cv2.magnitude(
            lo_dft_filtered[:, :, 0], lo_dft_filtered[:, :, 1]))

        magnitude_spectrum = np.log(cv2.magnitude(
            dft_shifted[:, :, 0], dft_shifted[:, :, 1]))

        # create 8-bit images to put the magnitude spectrum into

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
        hi_filtered_img_normalised = cv2.cvtColor(hi_filtered_img_normalised, cv2.COLOR_GRAY2BGR)
        lo_filtered_img_normalised = cv2.cvtColor(lo_filtered_img_normalised, cv2.COLOR_GRAY2BGR)
        hp_filter_vis = cv2.cvtColor(np.uint8(hp_filter[:, :, 0] * 255), cv2.COLOR_GRAY2BGR)
        lp_filter_vis = cv2.cvtColor(np.uint8(lp_filter[:, :, 0] * 255), cv2.COLOR_GRAY2BGR)

        # overlay corresponding labels on the images

        cv2.putText(gray_frame, 'Grayscale Input', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(magnitude_spectrum_normalised, f'Magnitude Spectrum', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(hi_filtered_img_normalised, f'High Pass Butterworth Output', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)        
        cv2.putText(lo_filtered_img_normalised, f'Low Pass Butterworth Output', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)   
        cv2.putText(hp_filter_vis, f'High Pass Butterworth Filter', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)        
        cv2.putText(lp_filter_vis, f'Low Pass Butterworth Filter', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)  

        # stack the images into a grid

        im_1 = cv2.hconcat([gray_frame, magnitude_spectrum_normalised])
        im_2 = cv2.hconcat([hp_filter_vis, hi_filtered_img_normalised])
        im_3 = cv2.hconcat([lp_filter_vis, lo_filtered_img_normalised])
        output = cv2.vconcat([im_1, im_2, im_3])

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
# Copyright (c) 2024 Dept Computer Science, Durham University, UK

# ===================================================================