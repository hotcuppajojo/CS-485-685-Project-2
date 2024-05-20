import numpy as np
from skimage.feature import match_descriptors
from image_processing.feature_extraction import extract_LBP, extract_HOG
from image_processing.corner_detection import moravec_detector, harris_detector

def feature_matching(image1, image2, detector, extractor):
    if image1.shape != image2.shape:
        raise ValueError('Input images must be of the same size.')

    if detector not in ['Moravec', 'Harris']:
        raise ValueError('Invalid detector. Must be "Moravec" or "Harris".')
    if extractor not in ['LBP', 'HOG']:
        raise ValueError('Invalid extractor. Must be "LBP" or "HOG".')

    keypoints1 = moravec_detector(image1) if detector == 'Moravec' else harris_detector(image1)
    keypoints2 = moravec_detector(image2) if detector == 'Moravec' else harris_detector(image2)
    if not keypoints1 or not keypoints2:
        return []

    descriptors1 = [extract_LBP(image1, kp) if extractor == 'LBP' else extract_HOG(image1, kp) for kp in keypoints1]
    descriptors2 = [extract_LBP(image2, kp) if extractor == 'LBP' else extract_HOG(image2, kp) for kp in keypoints2]

    descriptors1 = np.array(descriptors1).reshape(len(descriptors1), -1)
    descriptors2 = np.array(descriptors2).reshape(len(descriptors2), -1)

    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
    matched_pairs = [(keypoints1[match[0]], keypoints2[match[1]]) for match in matches]

    return matched_pairs
