
import sys
from pathlib import Path
# Add the parent directory of the script to the system path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))
import unittest
import numpy as np
from image_processing.feature_extraction import extract_LBP, extract_HOG

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.img_gray = np.zeros((100, 100), dtype=np.uint8)
        self.img_color = np.zeros((100, 100, 3), dtype=np.uint8)
        self.keypoint = (50, 50)

    def test_extract_LBP_gray(self):
        lbp = extract_LBP(self.img_gray, self.keypoint)
        self.assertIsNotNone(lbp)
        self.assertEqual(len(lbp), 8)

    def test_extract_LBP_color(self):
        lbp = extract_LBP(self.img_color, self.keypoint)
        self.assertIsNotNone(lbp)
        self.assertEqual(len(lbp), 24)

    def test_extract_HOG_gray(self):
        hog = extract_HOG(self.img_gray, self.keypoint)
        self.assertIsNotNone(hog)
        self.assertEqual(len(hog), 36)

    def test_extract_HOG_color(self):
        hog = extract_HOG(self.img_color, self.keypoint)
        self.assertIsNotNone(hog)
        self.assertEqual(len(hog), 108)

    def test_extract_LBP_invalid_keypoint(self):
        with self.assertRaises(IndexError):
            extract_LBP(self.img_gray, (150, 150))

    def test_extract_HOG_invalid_keypoint(self):
        with self.assertRaises(IndexError):
            extract_HOG(self.img_gray, (150, 150))

if __name__ == '__main__':
    unittest.main()