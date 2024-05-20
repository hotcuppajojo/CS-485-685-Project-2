import numpy as np
import cv2

def extract_LBP(image, keypoint, P=8, R=1):
    x, y = keypoint
    lbp = np.zeros((P,), dtype=int)
    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        lbp = []
        for channel in range(3):
            lbp_channel = extract_LBP_channel(image[:, :, channel], x, y, P, R)
            lbp.extend(lbp_channel)  # use extend instead of append
        lbp = np.array(lbp)  # convert list to numpy array
    else:  # grayscale image
        lbp = extract_LBP_channel(image, x, y, P, R)
    return lbp

def extract_LBP_channel(image, x, y, P, R):
    lbp = np.zeros((P,), dtype=int)
    for i in range(P):
        xi = x + R * np.cos(2 * np.pi * i / P)
        yi = y - R * np.sin(2 * np.pi * i / P)
        x0, y0 = int(np.floor(xi)), int(np.floor(yi))
        x1, y1 = int(np.ceil(xi)), int(np.ceil(yi))
        x0 = max(0, min(x0, image.shape[1] - 1))
        x1 = max(0, min(x1, image.shape[1] - 1))
        y0 = max(0, min(y0, image.shape[0] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        pixel = (image[y0, x0] * (x1 - xi) + image[y0, x1] * (xi - x0)) * (y1 - yi) + \
                (image[y1, x0] * (x1 - xi) + image[y1, x1] * (xi - x0)) * (yi - y0)
        lbp[i] = int(pixel > image[y, x])
    return lbp

def extract_HOG(image, keypoint, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    x, y = keypoint
    if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
        raise IndexError("Keypoint is out of bounds of the image")
    x_start = max(0, x - cell_size[1] * block_size[1] // 2)
    x_end = min(image.shape[1], x + cell_size[1] * block_size[1] // 2)
    y_start = max(0, y - cell_size[0] * block_size[0] // 2)
    y_end = min(image.shape[0], y + cell_size[0] * block_size[0] // 2)
    roi = image[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        return np.array([])

    if len(roi.shape) == 3 and roi.shape[2] == 3:  # color image
        hog = []
        for channel in range(3):
            roi_channel = roi[:, :, channel]
            gx = cv2.Sobel(roi_channel, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(roi_channel, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)
            bins = np.int32(nbins * ang / (2 * np.pi))
            hog_channel = []
            for i in range(block_size[0]):
                for j in range(block_size[1]):
                    bin_values = bins[i * cell_size[0]:(i + 1) * cell_size[0], j * cell_size[1]:(j + 1) * cell_size[1]]
                    hist, _ = np.histogram(bin_values, bins=nbins, range=(0, nbins))
                    eps = 1e-7
                    hist = hist.astype('float64')
                    hist /= np.sqrt(np.sum(hist**2) + eps**2)
                    hog_channel.append(hist)
            hog_channel = np.hstack(hog_channel)
            hog.append(hog_channel)
        hog = np.concatenate(hog)
    else:  # grayscale image
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(nbins * ang / (2 * np.pi))
        hog = []
        for i in range(block_size[0]):
            for j in range(block_size[1]):
                bin_values = bins[i * cell_size[0]:(i + 1) * cell_size[0], j * cell_size[1]:(j + 1) * cell_size[1]]
                hist, _ = np.histogram(bin_values, bins=nbins, range=(0, nbins))
                eps = 1e-7
                hist = hist.astype('float64')
                hist /= np.sqrt(np.sum(hist**2) + eps**2)
                hog.append(hist)
        hog = np.hstack(hog)
    return hog