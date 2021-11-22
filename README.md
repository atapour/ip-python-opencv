# Image Processing Teaching Examples (Python - OpenCV)

Image Processing examples used for teaching within the Department of Computer Science at [Durham University](https://www.durham.ac.uk/) (UK) by [Dr. Amir Atapour-Abarghouei](http://www.atapour.co.uk/).

The material is presented as part of the "Image Processing" lecture series at Durham University.

All material here has been tested with Opencv 4.5 and Python 3.9.

---

### Running the Code:

- You may download each file as needed.
- You can also download the entire repository as follows:

```
git clone https://github.com/atapour/ip-python-opencv
cd ip-python-opencv
```
In this repository, you can find:

+ .py file - python code for the examples

- You can simply run each Python file by running:

```
python <example file name>.py
```

---

#### - Arithmetic Point Transforms (```arithmetic_point_transforms.py```):
Running this script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input, converted greyscale, greyscale / 2, and the absolute difference between consecutive frames.

#### - Logical Point Transforms (```logical_point_transforms.py```):
Running this script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input, bitwise NOT of the converted greyscale, bitwise AND of the greyscale and binary circular mask, and the XOR of two consecutive frames.

#### - Logarithmic Transform (```logarithmic_transform.py```):
Running this script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input converted to greyscale and the logarithmic transform of the image. The parameters of the transform can be set using track bars.

#### - Exponential Transform (```exponential_transform.py```):
Running this script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input converted to greyscale and the exponential transform of the image. The parameters of the transform can be set using track bars.

#### - Power Law Transform - Gamma Correction (```gamma_correction.py```):
Running this script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input and the power law transform of the image [gamma correction]. The parameters of the transform can be set using track bars.

#### - Gaussian Noise Removal - Mean and Median Filtering (```gaussian_noise_removal.py```):
Running this script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input, Gaussian noise added to the input image, the mean filter applied to the image and the median filter applied to the image. The neighbourhood size of the filters can be set using the track bar.

#### - Salt and Pepper Noise Removal - Mean and Median Filtering (```salt_pepper_filter.py```):
Running this script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input, Salt and Pepper noise added to the input image, the mean filter applied to the image and the median filter applied to the image. The neighbourhood size of the filters can be set using the track bar.

#### - Laplacian Edge Sharpening (```laplacian.py```):
Running this script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input, Gaussian smoothing applied to the input image, the Laplacian of the image and the blurred image edge sharpened using the Laplacian.

#### - Bilateral Filtering (```bilateral_filter.py```):
Running this script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input, the Mean filter, the Gaussian filter and the Bilateral Filter applied to the image. The neighbourhood size of the mean and the Gaussian filters as well as the standard deviation of the Gaussian and the Bilateral Filters can be set using the track bar.

#### - Non-Local Means Filtering (```nlm_filter.py```):
Running this script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input with Salt and Pepper noise added to it, the Mean filter, the Gaussian filter and the Non-Local Mean Filter applied to the noisy image so the noise can be removed. The neighbourhood size of the mean and the Gaussian filters as well as the standard deviation of the Gaussian and the strength of the Non-Local Means Filters can be set using the track bar.

#### - Simple Contrast Stretching (```contrast_stretching.py```):
Running this script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input converted to grayscale, the histogram of the input, the output with its contrast stretched and the histogram of the contrast stretched output. 

#### - Contrast Equalisation (```equalise_histogram.py```):
Running this script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input converted to grayscale, the histogram of the input, the output with its histogram equalised and the histogram of the histogram equalised output. 

#### - CLAHE Equalisation (```equalise_clahe.py```):
Running this script will perform contrast limited adaptive histogram equalisation (CLAHE). The script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input converted to grayscale, the histogram of the input, the output after it is clahe equalised and the histogram of the clahe equalised output. Parameters of CLAHE equalisation can be set using track bars.

#### - Fourier Magnitude Spectrum (```fourier.py```):
Running this script will apply the Fourier Transform to an image and display the fourier magnitude spectrum. The script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input converted to grayscale, along with the Fourier magnitude spectrum of the image.

#### - Band-Pass Filtering (```bandpass-filter-fourier.py```):
Running this script will apply the Fourier Transform to an image and perform the band pass filtering. The script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input converted to grayscale, along with the mask that is meant to be applied to the Fourier magnitude spectrum of the image, the filter Fourier spectrum and the final filtered image brought back to the spatial domain.

#### - High/Low-Pass Filtering (```low-high-pass-filter-fourier.py```):
Running this script will apply the Fourier Transform to an image and perform both the high and low pass filtering. The script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original RGB, original input converted to grayscale, along with the high pass filter applied to the Fourier spectrum, the low pass filter applied to the Fourier spectrum, the final high pass filtered image brought back to the spatial domain and the final low pass filtered image brought back to the spatial domain. The radius of the filters can be determined using a track bar.

#### - Butterworth High/Low-Pass Filtering (```butterworth-low-high-pass-filter.py```):
Running this script will apply the Fourier Transform to an image and perform both the Butterworth high and low pass filtering. The script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input converted to grayscale and the Fourier spectrum along with the Butterworth high pass filter applied to the Fourier spectrum, the Butterworth low pass filter applied to the Fourier spectrum, the final high pass filtered image brought back to the spatial domain and the final low pass filtered image brought back to the spatial domain. The radius and order of the Butterworth filters can be set using track bars.

#### - Correlation - Template Matching (```correlation_template_matching.py```):
Running this script will apply template matching to images received from a camera (or a video). The user is asked draw a box on the image. This box, drawn using the mouse, selects a template. This template (patch to be matched) will be displayed as part of the output. Then correlation template matching will be performed, and a box will be drawn on the closest window to the input template within the image. This essentially means the algorithm will try to track the selected box.


#### - Visualising Colour Spaces (```colour-channels.py```):
Running this script will visualise the different colour channels in the RGB, HSV and CIELAB colour spaces. The script will separate the channels and display all three channels of all three colour spaces in a grid. Pushing the key 'c' will toggle colour mapping onto some of the colour channels.

#### - Colour Object Tracking (```colour-object-tracking.py```):
Running this script will perform colour object tracking on images received from a camera (or a video). The user is asked draw a box on the image. This box, drawn using the mouse, selects a patch. This patch will be displayed as part of the output. Then object tracking will be performed, and a box will be drawn on the closest window to the input patch within the image. This essentially means the algorithm will track the selected box. This tracking is based on the [Mean Shift](https://docs.opencv.org/3.4/d7/d00/tutorial_meanshift.html) Algorithm.


#### - Save Video (```save_video.py```):
Running this script will read a video from a camera and saves that video to disk. Most parameters are hard-coded and need to be changed in the code itself.

---

 ### Important Note:

All code is provided _"as is"_ to aid learning and understanding of topics within the "Image Processing" course.

---

Please raise an issue in this repository if you find any bugs.
It would even be better if you submitted a pull request with a fix or an improvement.
