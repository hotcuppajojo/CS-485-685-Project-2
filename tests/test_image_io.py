import sys
from pathlib import Path
# Add the parent directory of the script to the system path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))
import unittest
import time
import matplotlib.pyplot as plt
import numpy as np
from image_processing.image_io import load_img, display_img

class TestImageIO(unittest.TestCase):
    def test_load_img(self):
        img = load_img('test_images/3/test_img1.jpg', grayscale=True)
        self.assertIsNotNone(img)

    def test_load_img_invalid_file(self):
        with self.assertRaises(IOError):
            load_img('invalid_file.jpg')

    def test_load_img_color(self):
        img = load_img('test_images/3/test_img1.jpg', grayscale=False)
        self.assertEqual(img.ndim, 3)

    def test_display_img(self):
        img = load_img('test_images/3/test_img1.jpg')
        display_img(img)
        time.sleep(1)  # pause for 1 second
        plt.close('all')

    def test_display_img_none(self):
        with self.assertRaises(ValueError):
            display_img(None)

    def test_display_img_invalid_type(self):
        with self.assertRaises(TypeError):
            display_img('invalid_image')

    def test_display_img_invalid_dim(self):
        with self.assertRaises(ValueError):
            display_img(np.zeros((10, 10, 10, 10)))

if __name__ == '__main__':
    unittest.main()
