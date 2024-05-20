import cv2
import numpy as np

def moravec_detector(image, window_size=3, percentile=90):
    if image.size == 0:
        return []
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = []
    height, width = image.shape
    offset = window_size // 2
    responses = np.zeros((height, width))
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            min_response = float('inf')
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    window_orig = image[y-offset:y+offset+1, x-offset:x+offset+1]
                    y_shifted_start = max(0, y + dy - offset)
                    y_shifted_end = min(height, y + dy + offset + 1)
                    x_shifted_start = max(0, x + dx - offset)
                    x_shifted_end = min(width, x + dx + offset + 1)
                    window_shift = image[y_shifted_start:y_shifted_end, x_shifted_start:x_shifted_end]
                    if window_shift.shape != window_orig.shape:
                        continue
                    ssd = np.sum((window_orig - window_shift) ** 2)
                    min_response = min(min_response, ssd)
            responses[y, x] = min_response

    if np.count_nonzero(responses) == 0:
        return []

    threshold = np.percentile(responses[np.nonzero(responses)], percentile)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            if responses[y, x] > threshold:
                keypoints.append((x, y))
    return keypoints

def harris_detector(image, window_size=3, k=0.04):
    if image.size == 0:
        return []
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=window_size)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=window_size)
    Ixx = gaussian_filter(Ix**2, sigma=1)
    Ixy = gaussian_filter(Ix*Iy, sigma=1)
    Iyy = gaussian_filter(Iy**2, sigma=1)
    det = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    R = det - k * trace**2
    threshold = 0.01 * R.max()
    keypoints = np.argwhere(R > threshold)
    keypoints = [tuple(reversed(point)) for point in keypoints]
    return keypoints

def gaussian_filter(image, sigma=1):
    ksize = int(6*sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)
