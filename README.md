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
Running this script will perform contrast limited adaptive histogram equalisation(CLAHE). The script will receive input images from a camera or a video (pass the path to the video as an argument) and display the original input converted to grayscale, the histogram of the input, the output after it is clahe equalised and the histogram of the clahe equalised output. Parameters of CLAHE equalisation can be set using track bars.

---

 ### Important Note:

All code is provided _"as is"_ to aid learning and understanding of topics within the "Image Processing" course.

---

Please raise an issue in this repository if you find any bugs.
It would even be better if you submitted a pull request with a fix or an improvement.
