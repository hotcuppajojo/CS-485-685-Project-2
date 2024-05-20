import sys
from pathlib import Path
# Add the parent directory of the script to the system path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))
import unittest
import numpy as np
from feature_matching.matching import feature_matching
from image_processing.image_io import load_img

class TestFeatureMatching(unittest.TestCase):
    def setUp(self):
        self.img1 = load_img('test_images/3/test_img1.jpg', grayscale=True)
        self.img2 = load_img('test_images/3/test_img2.jpg', grayscale=True)
        self.img_empty = np.zeros((100, 100), dtype=np.uint8)

    def test_feature_matching_moravec_lbp(self):
        matched_pairs = feature_matching(self.img1, self.img2, 'Moravec', 'LBP')
        self.assertIsNotNone(matched_pairs)

    def test_feature_matching_harris_hog(self):
        matched_pairs = feature_matching(self.img1, self.img2, 'Harris', 'HOG')
        self.assertIsNotNone(matched_pairs)

    def test_feature_matching_invalid_detector(self):
        with self.assertRaises(ValueError):
            feature_matching(self.img1, self.img2, 'InvalidDetector', 'LBP')

    def test_feature_matching_invalid_extractor(self):
        with self.assertRaises(ValueError):
            feature_matching(self.img1, self.img2, 'Moravec', 'InvalidExtractor')

    def test_feature_matching_no_keypoints(self):
        matched_pairs = feature_matching(self.img_empty, self.img_empty, 'Moravec', 'LBP')
        self.assertEqual(matched_pairs, [])

    def test_feature_matching_different_size_images(self):
        img_small = self.img1[:50, :50]
        with self.assertRaises(ValueError):
            feature_matching(img_small, self.img2, 'Moravec', 'LBP')

    def test_feature_matching_color_images(self):
        img_color1 = np.stack([self.img1]*3, axis=-1)
        img_color2 = np.stack([self.img2]*3, axis=-1)
        matched_pairs = feature_matching(img_color1, img_color2, 'Moravec', 'LBP')
        self.assertIsNotNone(matched_pairs)

    def test_feature_matching_same_image(self):
        matched_pairs = feature_matching(self.img1, self.img1, 'Moravec', 'LBP')
        self.assertIsNotNone(matched_pairs)

if __name__ == '__main__':
    unittest.main()