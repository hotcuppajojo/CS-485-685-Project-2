import sys
from pathlib import Path
# Add the parent directory of the script to the system path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))
import unittest
import numpy as np
from image_processing.corner_detection import moravec_detector, harris_detector
from image_processing.image_io import load_img

class TestCornerDetection(unittest.TestCase):
    def setUp(self):
        # Load the test image
        self.image = load_img('test_images/3/test_img1.jpg', grayscale=True)
        self.color_image = load_img('test_images/3/test_img1.jpg', grayscale=False)  # Load color image
        # Create a binary image
        self.binary_image = np.zeros((100, 100), dtype=np.uint8)
        self.binary_image[50:60, 50:60] = 255  # Add a white square in the middle
        # Create an empty image
        self.empty_image = np.zeros((0, 0), dtype=np.uint8)
        # Create an image with no corners
        self.no_corners_image = np.ones((100, 100), dtype=np.uint8) * 255
        # Create an image with only corners
        self.only_corners_image = np.zeros((100, 100), dtype=np.uint8)
        self.only_corners_image[0, :] = 255
        self.only_corners_image[:, 0] = 255
        self.only_corners_image[-1, :] = 255
        self.only_corners_image[:, -1] = 255
        self.only_corners_image[25:75, 25:75] = 255
        self.only_corners_image[35:65, 35:65] = 0

    def test_moravec_detector(self):
        # Test with grayscale image
        print("Testing with grayscale image")
        keypoints = moravec_detector(self.image, percentile=75)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

        # Test with color image
        print("Testing with color image")
        keypoints = moravec_detector(self.color_image, percentile=75)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

        # Test with binary image
        print("Testing with binary image")
        keypoints = moravec_detector(self.binary_image, percentile=75)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

        # Test with different percentile values
        print("Testing with different percentile values")
        keypoints = moravec_detector(self.image, percentile=50)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

        # Test with empty image
        print("Testing with empty image")
        keypoints = moravec_detector(self.empty_image, percentile=75)
        self.assertEqual(len(keypoints), 0)

        # Test with image of different size
        print("Testing with image of different size")
        large_image = np.resize(self.image, (200, 200))
        keypoints = moravec_detector(large_image, percentile=75)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

        # Test with image that contains no corners
        print("Testing with image that contains no corners")
        keypoints = moravec_detector(self.no_corners_image, percentile=75)
        self.assertEqual(len(keypoints), 0)

        # Test with image that contains only corners
        print("Testing with image that contains only corners")
        keypoints = moravec_detector(self.only_corners_image, window_size=5, percentile=75)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

    def test_harris_detector(self):
        # Test with grayscale image
        keypoints = harris_detector(self.image, window_size=3, k=0.04)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

        # Test with color image
        keypoints = harris_detector(self.color_image, window_size=3, k=0.04)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

        # Test with binary image
        keypoints = harris_detector(self.binary_image, window_size=3, k=0.04)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

        # Test with different window sizes
        keypoints = harris_detector(self.image, window_size=5, k=0.04)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

        # Test with different k values
        keypoints = harris_detector(self.image, window_size=3, k=0.06)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

        # Test with empty image
        keypoints = harris_detector(self.empty_image, window_size=3, k=0.04)
        self.assertEqual(len(keypoints), 0)

        # Test with image of different size
        large_image = np.resize(self.image, (200, 200))
        keypoints = harris_detector(large_image, window_size=3, k=0.04)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

        # Test with image that contains no corners
        keypoints = harris_detector(self.no_corners_image, window_size=3, k=0.04)
        self.assertEqual(len(keypoints), 0)

        # Test with image that contains only corners
        keypoints = harris_detector(self.only_corners_image, window_size=3, k=0.04)
        self.assertGreater(len(keypoints), 0)
        self._check_keypoints_within_bounds(keypoints)

    def _check_keypoints_within_bounds(self, keypoints):
        for keypoint in keypoints:
            self.assertGreaterEqual(keypoint[0], 0)
            self.assertGreaterEqual(keypoint[1], 0)
            self.assertLess(keypoint[0], self.image.shape[1])
            self.assertLess(keypoint[1], self.image.shape[0])

if __name__ == "__main__":
    unittest.main()