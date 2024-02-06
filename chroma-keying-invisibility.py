# ===================================================================

# Example : perform chroma-keying to create the cloak of invisibility
# this includes chroma keying as well as a convex hull operation 
# around the foreground and feathered blending for compositing

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2024 Amir Atapour Abarghouei

# based on : https://github.com/atapour/invisible-me
# License : MIT - https://opensource.org/license/mit/

# ===================================================================

import cv2
import numpy as np
import warnings
import argparse

# ===================================================================

warnings.filterwarnings("ignore")
keep_processing = True

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(
    description='Cloak of Invisibility.')

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

# define the range of hues to detect - set automatically using mouse

lower_bound = np.array([255, 0, 0])
upper_bound = np.array([255, 255, 255])

# ===================================================================

# mouse callback function - activated on any mouse event (click, movement)
# displays and sets Hue range based on right click location


def mouse_callback(event, x, y, flags, param):

    global upper_bound
    global lower_bound

    # records mouse events at position (x,y) in the image window
    # left button click prints colour HSV information and sets range

    if event == cv2.EVENT_LBUTTONDOWN:
        print("HSV colour @ position (%d,%d) = %s (bounds set with +/- 20)" %
              (x, y, ', '.join(str(i) for i in image_hsv[y, x])))

        # set Hue bounds on the Hue with +/- 15 threshold on the range

        upper_bound[0] = image_hsv[y, x][0] + 20
        lower_bound[0] = image_hsv[y, x][0] - 20

        # set Saturation and Value to eliminate very dark, noisy image regions

        lower_bound[1] = 50
        lower_bound[2] = 50

    # right button click resets HSV range

    elif event == cv2.EVENT_RBUTTONDOWN:

        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([255, 255, 255])


# ===================================================================

# define video capture with access to camera defined by input arg

print("Starting camera stream!")

camera = cv2.VideoCapture(args.camera)

# define display window

window_name = "Live Camera Input - Improved Cloak of Invisibility"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# set the mouse call back function that will be called every time
# the mouse is clicked inside the display window

cv2.setMouseCallback(window_name, mouse_callback)

# ===================================================================

# first, take an image of the background image
# we will skip some frames first to capture a clean one

for i in range(10):
    _, background = camera.read()

# ===================================================================

keep_processing = True

while (keep_processing):

    # read an image from the camera

    _, image = camera.read()

    # start a timer (to see how long processing and display takes)

    start_t = cv2.getTickCount()

    # convert the RGB images to HSV

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # create a foreground mask that identifies the pixels in the range of hues

    foreground_mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

    # logically invert the foreground mask to get the background mask via NOT

    background_mask = cv2.bitwise_not(foreground_mask)

    # perform morphological opening and dilation on the foreground mask

    foreground_mask_morphed = cv2.morphologyEx(foreground_mask,
                                               cv2.MORPH_OPEN,
                                               np.ones((3, 3), np.uint8),
                                               iterations=5)
    foreground_mask_morphed = cv2.morphologyEx(foreground_mask_morphed,
                                               cv2.MORPH_DILATE,
                                               np.ones((3, 3), np.uint8),
                                               iterations=5)

    # extract the set of contours around the foreground mask and then the
    # convex hull around that set of contours. Update the foreground mask with
    # the convex hull of all the pixels in the region

    contours, _ = cv2.findContours(foreground_mask_morphed,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) > 0):
        hull = cv2.convexHull(np.vstack(list(contours[i]
                              for i in range(len(contours)))))
        cv2.fillPoly(foreground_mask_morphed, [hull], (255, 255, 255))

    # logically invert the foreground mask to get the background mask via NOT

    background_mask = cv2.bitwise_not(foreground_mask_morphed)

    # construct 3-channel RGB feathered background mask for blending

    foreground_mask_feathered = cv2.blur(foreground_mask_morphed,
                                         (15, 15)) / 255.0
    background_mask_feathered = cv2.blur(background_mask, (15, 15)) / 255.0
    background_mask_feathered = cv2.merge([background_mask_feathered,
                                           background_mask_feathered,
                                           background_mask_feathered])
    foreground_mask_feathered = cv2.merge([foreground_mask_feathered,
                                           foreground_mask_feathered,
                                           foreground_mask_feathered])

    # combine current camera image with cloaked region via feathered blending

    cloaked_image = ((background_mask_feathered * image) +
                     (foreground_mask_feathered * background)).astype('uint8')

    # resize background image for visualisation

    background_visual = cv2.resize(background,
                                   (int(background.shape[1] * 0.2),
                                    int(background.shape[0] * 0.2)),
                                   interpolation=cv2.INTER_AREA)

    # label the background image for visualisation

    cv2.putText(background_visual, "Background",
                (10, background_visual.shape[0] - 15),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (123, 49, 126))

    # overlay the background image on the cloaked frame

    background_width = background_visual.shape[1]
    background_height = background_visual.shape[0]
    image_width = image.shape[1]
    image_height = image.shape[0]

    cloaked_image[
        0:background_height,
        (image_width - background_width):image_width
        ] = background_visual

    # stop the timer and convert to milliseconds
    # (to see how long processing and display takes)

    stop_t = ((cv2.getTickCount() - start_t) /
              cv2.getTickFrequency()) * 1000

    label = ('Processing time: %.2f ms' % stop_t) + \
        (' (Max Frames per Second (fps): %.2f' % (1000 / stop_t)) + ')'
    cv2.putText(cloaked_image, label, (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # overlay labels

    label1 = "click on the (green) chroma keying material"
    label2 = f'HSV colour range: {lower_bound} - {upper_bound}'
    label3 = "press space to re-capture background"

    cv2.putText(cloaked_image, label1, (10, cloaked_image.shape[0] - 85),
                cv2.FONT_HERSHEY_COMPLEX, 1, (123, 49, 126), 3)

    cv2.putText(cloaked_image, label2, (10, cloaked_image.shape[0] - 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (123, 49, 126), 3)

    cv2.putText(cloaked_image, label3, (10, cloaked_image.shape[0] - 15),
                cv2.FONT_HERSHEY_COMPLEX, 1, (123, 49, 126), 3)

    # display image with cloaking present

    cv2.imshow(window_name, cloaked_image)

    # start the event loop - if user presses "q" then exit
    # wait 40ms for a key press from the user (i.e. 1000ms / 25 fps = 40 ms)

    key = cv2.waitKey(40) & 0xFF

    # - if user presses q then exit

    if (key == ord('q')):
        keep_processing = False

    # - if user presses space then reset background

    elif (key == ord(' ')):
        print("\n -- resetting background image")
        _, background = camera.read()

# ===================================================================

# Author : Amir Atapour-Abarghouei
# Copyright (c) 2024 Dept Computer Science, Durham University, UK

# ===================================================================
