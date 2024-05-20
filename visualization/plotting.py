import cv2
import numpy as np

def plot_keypoints(image, keypoints):
    if 0 in image.shape or image.size == 0:
        raise ValueError("The input image is empty.")
    image_copy = image.copy()
    if len(image_copy.shape) == 3:
        image_rgb = image_copy
    else:
        image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
    for keypoint in keypoints:
        if not isinstance(keypoint[0], int) or not isinstance(keypoint[1], int):
            raise ValueError("Keypoints must be integers.")
        keypoint = tuple(map(int, keypoint))
        if not (0 <= keypoint[0] < image.shape[1] and 0 <= keypoint[1] < image.shape[0]):
            raise ValueError("Invalid keypoint.")
        cv2.circle(image_rgb, keypoint, 1, (255, 0, 0), -1)
    return image_rgb

def plot_matches(image1, image2, matches):
    if 0 in image1.shape or 0 in image2.shape or image1.size == 0 or image2.size == 0:
        raise ValueError("One or both of the input images are empty.")
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

    new_image = np.zeros((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3), dtype="uint8")
    new_image[:image1.shape[0], :image1.shape[1]] = image1
    new_image[:image2.shape[0], image1.shape[1]:] = image2
    offset_x = image1.shape[1]
    for match in matches:
        if not (isinstance(match[0][0], int) and isinstance(match[0][1], int) and
                isinstance(match[1][0], int) and isinstance(match[1][1], int)):
            raise ValueError("Match coordinates must be integers.")
        pt1, pt2 = tuple(map(int, match[0])), tuple(map(int, (match[1][0] + offset_x, match[1][1])))
        print(f"Checking match: pt1={pt1}, pt2={pt2}")
        if not (0 <= pt1[0] < image1.shape[1] and 0 <= pt1[1] < image1.shape[0] and
                0 <= pt2[0] - offset_x < image2.shape[1] and 0 <= pt2[1] < image2.shape[0]):
            print(f"Invalid match detected: pt1: {pt1}, pt2: {pt2}, image1.shape: {image1.shape}, image2.shape: {image2.shape}")
            raise ValueError("Invalid match.")
        cv2.circle(new_image, pt1, 5, (0, 0, 255), -1)
        cv2.circle(new_image, pt2, 5, (0, 0, 255), -1)
        cv2.line(new_image, pt1, pt2, (0, 0, 255), 1)
    return new_image