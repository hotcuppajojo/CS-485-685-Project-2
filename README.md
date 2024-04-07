# CS 485/685 - Project 2: Image Filtering
**Author:** JoJo Petersky  
**Date:** 2024/04/07  
**Course:** CS 485/685 Spring '24  

## Introduction
This project explores the implementation of various image filtering techniques, such as interest point detection, feature extraction, and feature matching, for both color and grayscale images. Utilizing Python and libraries like NumPy, OpenCV, Matplotlib, and others, the project aims to apply these techniques effectively. This README file provides insights into the project structure, implementation specifics, and guidelines for usage.

## Project Structure
Contained within a single Python file, `project2.py`, this project encapsulates all the requisite functions for the assignment. These include functionalities for loading and displaying images, plotting keypoints, extracting Local Binary Pattern (LBP) and Histogram of Oriented Gradients (HOG) features, matching features between images, and visualizing matches.

## Setup and Execution
Ensure Python 3.x is installed on your system, along with these necessary packages:
- `numpy`
- `opencv-python` (cv2)
- `matplotlib`
- `skimage`

Use pip to install the packages:
```
pip install numpy opencv-python matplotlib scikit-image
```

To run the script, execute:
```
python project2.py
```

## Implementation Details
### 1. Load and Display Images
- **`load_img(file_name, grayscale=False)`**: Loads an image from a specified path. If `grayscale` is True, the image is loaded in grayscale mode; otherwise, it's loaded in color.
- **`display_img(image)`**: Displays the provided image, supporting both color and grayscale formats.

### 2. Moravec Detector
- **`moravec_detector(image)`**: Implements the Moravec corner detection algorithm to detect and return a list of keypoints within an image.

### 3. Plot Keypoints
- **`plot_keypoints(image, keypoints)`**: Marks keypoints on an image with red, returning a new image with these markings.

### 4. LBP Features
- **`extract_LBP(image, keypoint)`**: Extracts Local Binary Pattern (LBP) features from a specified keypoint in an image.

### 5. HOG Features
- **`extract_HOG(image, keypoint)`**: Derives Histogram of Oriented Gradients (HOG) features from a given keypoint in an image.

### 6. Feature Matching
- **`feature_matching(image1, image2, detector, extractor)`**: Matches features between two images, using either `Moravec` or `Harris` as the detector and `LBP` or `HOG` as the extractor.

### 7. Plot Matches
- **`plot_matches(image1, image2, matches)`**: Visualizes matching keypoints between two images with red markers for keypoints and red lines for matches.

## Additional Notes
- The project supports processing both color and grayscale images. It adapts feature extraction and other functionalities based on the image type.
- Utilize `test_script.py` to test the functions' correctness. It's designed to work well if your implementation adheres to the specifications.
- For comprehensive details on the algorithms and their implementation, refer to the in-depth comments within `project2.py`.

## License
This project is intended for educational use and is the intellectual property of JoJo Petersky and the instructors of CS 485/685.