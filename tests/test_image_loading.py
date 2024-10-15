import os
import sys
import unittest
import cv2 as cv
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import load_img, convert_to_grayscale, apply_th

class TestImageLoading(unittest.TestCase):
    def setUp(self):
        # Set up image path
        self.imgPath = 'C:/Users/Darien/Downloads/RefImg.png'
        self.testImg = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    def test_load_img(self):
        # Check if the image file exists
        self.assertTrue(os.path.exists(self.imgPath), "Image File Not Found")
        image = load_img(self.imgPath)
        self.assertIsNotNone(image, "Failed to Load Image")
    
    def test_convert_to_grayscale(self):
        grayscale_image = convert_to_grayscale(self.testImg)
        self.assertEqual(len(grayscale_image.shape), 2, "Grayscale Image Shape Mismatch")

    def test_apply_th(self):
        grayscale_image = convert_to_grayscale(self.testImg)
        th_image = apply_th(grayscale_image)
        self.assertEqual(len(th_image.shape), 2, "Thresholded Image Shape Mismatch")
        self.assertTrue(np.array_equal(np.unique(th_image), [0, 255]), "Thresholding failed")

if __name__ == "__main__":
    unittest.main()
