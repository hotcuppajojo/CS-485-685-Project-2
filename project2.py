# JoJo Petersky
# CS 485/685 Spring '24 Project2
# 2024/04/07
# project2.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_descriptors

# Image Loading & Displaying Functions
# ------------------------------------    

def load_img(file_name, grayscale=False):
    """
    Loads an image from a file.

    Parameters:
    - file_name: The path to the image file.
    - grayscale: A boolean flag indicating whether to load the image as grayscale.

    Returns:
    - The loaded image as a numpy array.

    Raises:
    - IOError: If the image cannot be loaded.
    """
    try:
        # Choose the color mode based on the grayscale flag
        color_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

        # Attempt to load the image
        img = cv2.imread(file_name, color_mode)

        # Check if the image was loaded successfully
        if img is None:
            raise ValueError(f"The file {file_name} cannot be loaded as an image.")
        
        return img
    except Exception as e:
        # Handle other potential exceptions (e.g., file not found, no read permissions)
        raise IOError(f"An error occurred when trying to load the image: {e}")
    
def display_img(image):
    """
    Displays an image using OpenCV's imshow function.

    Parameters:
    - image: A numpy ndarray representing the image to be displayed.

    Raises:
    - ValueError: If the input image is None.
    - TypeError: If the input is not a numpy ndarray.
    - ValueError: If the input is not a 2D (grayscale) or 3D (color) image.
    """
    if image is None:
        raise ValueError("No image to display. Image input is None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")

    if image.ndim not in [2, 3]:
        raise ValueError("Input must be a 2D (grayscale) or 3D (color) image.")

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Moravec Corner Detector
# ------------------------

def moravec_detector(image):
    # Initialize keypoints list
    keypoints = []

    # Get image dimensions
    height, width = image.shape

    # Define window size
    window_size = 3
    offset = window_size // 2

    # Compute Moravec response
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Initialize minimum response value
            min_response = float('inf')

            # Compute response for each direction
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    # Compute sum of squared differences
                    ssd = np.sum((image[y - offset:y + offset + 1, x - offset:x + offset + 1] - 
                                  image[y - offset + dy:y + offset + dy + 1, x - offset + dx:x + offset + dx + 1]) ** 2)

                    # Update minimum response value
                    min_response = min(min_response, ssd)

            # If the response is above a threshold, add the point to the keypoints list
            if min_response > 10000:
                keypoints.append((x, y))

    return keypoints

# Keypoint Plotting Function
# --------------------------

def plot_keypoints(image, keypoints):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Plot keypoints
    for keypoint in keypoints:
        cv2.circle(image_rgb, keypoint, 1, (255, 0, 0), -1)

    return image_rgb

# Local Binary Pattern (LBP) Descriptor
# -------------------------------------

def extract_LBP(image, keypoint):
    # Define the LBP parameters
    P = 8  # Number of circularly symmetric neighbour set points (quantization of the angular space)
    R = 1  # Radius of circle (spatial resolution of the operator)

    # Get the coordinates of the keypoint
    x, y = keypoint

    # Initialize the LBP feature vector
    lbp = np.zeros((P,), dtype=int)

    # Compute the LBP code
    for i in range(P):
        # Compute the coordinates of the circularly symmetric neighbour
        xi = x + R * np.cos(2 * np.pi * i / P)
        yi = y - R * np.sin(2 * np.pi * i / P)

        # Perform bilinear interpolation to get the pixel value
        x0, y0 = int(np.floor(xi)), int(np.floor(yi))
        x1, y1 = int(np.ceil(xi)), int(np.ceil(yi))
        pixel = (image[y0, x0] * (x1 - xi) + image[y0, x1] * (xi - x0)) * (y1 - yi) + \
                (image[y1, x0] * (x1 - xi) + image[y1, x1] * (xi - x0)) * (yi - y0)

        # Update the LBP code
        lbp[i] = int(pixel > image[y, x])

    # Convert the LBP code to a single number
    lbp = np.dot(2**np.arange(P), lbp)

    return lbp

# Histogram of Oriented Gradients (HOG) Descriptor
# ------------------------------------------------

def extract_HOG(image, keypoint, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    # Get the coordinates of the keypoint
    x, y = keypoint

    # Define the region of interest around the keypoint
    roi = image[y - cell_size[0] * block_size[0] // 2:y + cell_size[0] * block_size[0] // 2,
                x - cell_size[1] * block_size[1] // 2:x + cell_size[1] * block_size[1] // 2]

    # Compute the gradients in the x and y directions
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1)

    # Compute the gradient magnitude and orientation
    mag, ang = cv2.cartToPolar(gx, gy)

    # Quantize the orientation to the specified number of bins
    bins = np.int32(nbins * ang / (2 * np.pi))

    # Initialize the HOG feature vector
    hog = []

    # Compute the HOG features for each cell in the block
    for i in range(block_size[0]):
        for j in range(block_size[1]):
            # Get the bin values for the current cell
            bin_values = bins[i * cell_size[0]:(i + 1) * cell_size[0], j * cell_size[1]:(j + 1) * cell_size[1]]

            # Compute the histogram for the current cell
            hist, _ = np.histogram(bin_values, bins=nbins, range=(0, nbins))

            # Normalize the histogram
            eps = 1e-7  # To avoid division by zero
            hist /= np.sqrt(np.sum(hist**2) + eps**2)

            # Append the histogram to the HOG feature vector
            hog.append(hist)

    # Flatten the HOG feature vector
    hog = np.hstack(hog)

    return hog

# Feature Matching Functions
# ------------------------------------------------

def feature_matching(image1, image2, detector, extractor):
    # Validate detector
    if detector not in ['Moravec', 'Harris']:
        raise ValueError('Invalid detector. Must be "Moravec" or "Harris".')

    # Validate extractor
    if extractor not in ['LBP', 'HOG']:
        raise ValueError('Invalid extractor. Must be "LBP" or "HOG".')

    # Call appropriate detector
    if detector == 'Moravec':
        keypoints1 = moravec_detector(image1)
        keypoints2 = moravec_detector(image2)
    else:  # Harris
        keypoints1 = harris_detector(image1)
        keypoints2 = harris_detector(image2)

    # Call appropriate extractor
    if extractor == 'LBP':
        descriptors1 = extract_LBP(image1, keypoints1)
        descriptors2 = extract_LBP(image2, keypoints2)
    else:  # HOG
        descriptors1 = extract_HOG(image1, keypoints1)
        descriptors2 = extract_HOG(image2, keypoints2)

    # Match descriptors
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    # Return lists of matching keypoints
    match_keypoints1 = [keypoints1[match[0]] for match in matches]
    match_keypoints2 = [keypoints2[match[1]] for match in matches]

    return match_keypoints1, match_keypoints2

def plot_matches(image1, image2, matches):
    # Create a new image that can fit both input images
    new_image = np.zeros((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3), dtype="uint8")

    # Place the input images in the new image
    new_image[:image1.shape[0], :image1.shape[1]] = image1
    new_image[:image2.shape[0], image1.shape[1]:] = image2

    for match in matches:
        # Draw the match
        pt1 = (int(match[0][0]), int(match[0][1]))
        pt2 = (int(match[1][0]) + image1.shape[1], int(match[1][1]))
        cv2.circle(new_image, pt1, 5, (0, 0, 255), -1)
        cv2.circle(new_image, pt2, 5, (0, 0, 255), -1)
        cv2.line(new_image, pt1, pt2, (0, 0, 255), 1)

    return new_image

def harris_detector(image, window_size=3, k=0.04):
    # Convert to grayscale if necessary
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute image gradients
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    # Compute products of gradients and squares of gradients
    Ixx = gaussian_filter(Ix**2, sigma=window_size)
    Ixy = gaussian_filter(Ix*Iy, sigma=window_size)
    Iyy = gaussian_filter(Iy**2, sigma=window_size)

    # Compute determinant and trace
    det = Ixx * Iyy - Ixy ** 2
    trace = Ixx + Iyy

    # Compute response of Harris detector
    R = det - k * trace ** 2

    # Find local maxima
    keypoints = np.argwhere(R > np.max(R) * 0.01)

    return keypoints

# Gaussian Filter Function
# ------------------------
    
def gaussian_filter(image, sigma, filter_w, filter_h, pad_pixels, pad_value):
    """
    Applies a Gaussian filter to an image.

    Parameters:
    - image: The input image as a 2D numpy array.
    - sigma: Standard deviation of the Gaussian distribution.
    - filter_w: Width of the filter kernel.
    - filter_h: Height of the filter kernel.
    - pad_pixels: The amount of padding to apply to the image.
    - pad_value: The value to use for padding.

    Returns:
    - The filtered image as a 2D numpy array.
    """
    # Generate Gaussian filter
    m, n = [(ss-1.)/2. for ss in (filter_w, filter_h)]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    filter = h

    stride = 1  # This is the stride of your convolution
    filter_height, filter_width = filter.shape

    # Calculate padding for height and width separately
    pad_height = pad_pixels if isinstance(pad_pixels, int) else pad_pixels[0]
    pad_width = pad_pixels if isinstance(pad_pixels, int) else pad_pixels[1]

    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)),
                          mode='constant' if pad_value == 0 else 'edge',
                          constant_values=pad_value)

    # Calculate the dimensions of the output image
    output_height = ((image.shape[0] - filter_height + 2 * pad_height) // stride) + 1
    output_width = ((image.shape[1] - filter_width + 2 * pad_width) // stride) + 1
    output = np.zeros((output_height, output_width))

    # Apply the filter
    for i in range(0, output_height):
        for j in range(0, output_width):
            # Define the current region of interest
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + filter_height
            end_j = start_j + filter_width
            window = padded_image[start_i:end_i, start_j:end_j]

            # Calculate the convolution result (element-wise multiplication and sum)
            output_value = np.sum(window * filter)

            # Store the result
            output[i, j] = output_value

    # Normalize the output to the range [0, 255]
    output = ((output - np.min(output)) / (np.max(output) - np.min(output)) * 255)
    output = np.clip(output, 0, 255).astype(np.uint8)

    # Return the output
    return output