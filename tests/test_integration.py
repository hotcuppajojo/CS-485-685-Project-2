import sys
from pathlib import Path
# Add the parent directory of the script to the system path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))
import unittest
from image_processing.image_io import load_img, display_img
from image_processing.corner_detection import moravec_detector, harris_detector
from image_processing.feature_extraction import extract_LBP, extract_HOG
from feature_matching.matching import feature_matching
from visualization.plotting import plot_keypoints, plot_matches

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.grayscale_img1 = load_img('test_images/3/test_img1.jpg', grayscale=True)
        self.grayscale_img2 = load_img('test_images/3/test_img2.jpg', grayscale=True)

    def test_integration_workflow(self):
        # Test image load and display
        self.assertIsNotNone(self.grayscale_img1)
        display_img(self.grayscale_img1)

        # Test corner detection and plotting
        keypoints = moravec_detector(self.grayscale_img1)
        img_with_keypoints = plot_keypoints(self.grayscale_img1, keypoints)
        self.assertIsNotNone(img_with_keypoints)

        # Test LBP workflow
        lbps = [extract_LBP(self.grayscale_img1, kp) for kp in keypoints]
        self.assertIsNotNone(lbps)

        # Test HOG workflow
        hogs = [extract_HOG(self.grayscale_img1, kp) for kp in keypoints]
        self.assertIsNotNone(hogs)

        # Test feature matching and visualization
        matches = feature_matching(self.grayscale_img1, self.grayscale_img2, 'Moravec', 'LBP')
        img_with_matches = plot_matches(self.grayscale_img1, self.grayscale_img2, matches)
        self.assertIsNotNone(img_with_matches)
        
if __name__ == '__main__':
    unittest.main()