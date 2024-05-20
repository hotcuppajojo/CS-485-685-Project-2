import sys
from pathlib import Path
# Add the parent directory of the script to the system path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))
import unittest
import numpy as np
from visualization.plotting import plot_keypoints, plot_matches
from image_processing.image_io import load_img

class TestPlotting(unittest.TestCase):
    def setUp(self):
        # Load the test grayscale images
        self.grayscale_img1 = load_img('test_images/3/test_img1.jpg', grayscale=True)
        self.grayscale_img2 = load_img('test_images/3/test_img2.jpg', grayscale=True)
        # Load the test color images
        self.color_img1 = load_img('test_images/3/test_img1.jpg', grayscale=False)
        self.color_img2 = load_img('test_images/3/test_img2.jpg', grayscale=False)
        # Create keypoints
        self.keypoints = [(50, 50), (60, 60)]
        # Create matches
        self.matches = [((10, 10), (20, 20)), ((30, 30), (40, 40))]
        # Create images of different sizes
        self.img1_diff_size = np.zeros((100, 100, 3), dtype=np.uint8)
        self.img2_diff_size = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create an empty image
        self.empty_img = np.empty((0, 0), dtype=np.uint8)

    def test_plot_keypoints(self):
        # Test with grayscale image
        img_with_keypoints = plot_keypoints(self.grayscale_img1, self.keypoints)
        self.assertIsNotNone(img_with_keypoints)

        # Test with color image
        img_with_keypoints = plot_keypoints(self.color_img1, self.keypoints)
        self.assertIsNotNone(img_with_keypoints)

        # Test with no keypoints
        img_with_keypoints = plot_keypoints(self.grayscale_img1, [])
        self.assertIsNotNone(img_with_keypoints)

    def test_plot_matches(self):
        # Test with grayscale images
        img_with_matches = plot_matches(self.grayscale_img1, self.grayscale_img2, self.matches)
        self.assertIsNotNone(img_with_matches)

        # Test with color images
        img_with_matches = plot_matches(self.color_img1, self.color_img2, self.matches)
        self.assertIsNotNone(img_with_matches)

        # Test with no matches
        img_with_matches = plot_matches(self.color_img1, self.color_img2, [])
        self.assertIsNotNone(img_with_matches)

        # Test with images of different sizes
        img_with_matches = plot_matches(self.img1_diff_size, self.img2_diff_size, self.matches)
        self.assertIsNotNone(img_with_matches)
    
    def test_plot_keypoints_with_edge_cases(self):
        # Test with empty image
        with self.assertRaises(ValueError):
            plot_keypoints(self.empty_img, self.keypoints)

        # Test with invalid keypoints
        invalid_keypoints = [(1000, 1000), (-10, -10)]
        with self.assertRaises(ValueError):
            plot_keypoints(self.grayscale_img1, invalid_keypoints)

        # Test with non-integer keypoints
        non_integer_keypoints = [(50.5, 50.5), (60.5, 60.5)]
        with self.assertRaises(ValueError):
            plot_keypoints(self.grayscale_img1, non_integer_keypoints)

    def test_plot_matches_with_edge_cases(self):
        # Manually raise ValueError to ensure it's caught
        with self.assertRaises(ValueError):
            raise ValueError("Manual error")

        # Test with empty images
        with self.assertRaises(ValueError):
            plot_matches(self.empty_img, self.empty_img, self.matches)

        # Test with invalid matches
        invalid_matches = [((1000, 1000), (1000, 1000)), ((-10, -10), (-10, -10))]
        with self.assertRaises(ValueError):
            plot_matches(self.grayscale_img1, self.grayscale_img2, invalid_matches)

        # Test with non-integer matches
        non_integer_matches = [((50.5, 50.5), (60.5, 60.5)), ((70.5, 70.5), (80.5, 80.5))]
        with self.assertRaises(ValueError):
            plot_matches(self.grayscale_img1, self.grayscale_img2, non_integer_matches)


if __name__ == "__main__":
    unittest.main()
