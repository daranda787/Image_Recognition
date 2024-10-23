import os
import tempfile
import sys
import unittest
import cv2 as cv
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from image_processor import load_img, convert_to_grayscale, apply_threshold, process_image, process_all_images, display_image, process_image

DISPLAY_IMAGES = False
class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        # Set up image path
        self.imgPath = 'C:/Users/Darien/Downloads/RefImg.png'
        self.testImg = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    def tearDown(self):
        # Clean up the test image after each test
        self.testImg = None

    # Test if the image file exists 
    def test_load_img(self):
        # Check if the image file exists
        self.assertTrue(os.path.exists(self.imgPath), "Image File Not Found")
        image = load_img(self.imgPath)
        self.assertIsNotNone(image, "Failed to Load Image")
    
    def test_convert_to_grayscale(self):
        grayscale_image = convert_to_grayscale(self.testImg)
        self.assertEqual(len(grayscale_image.shape), 2, "Grayscale Image Shape Mismatch")

    def test_apply_threshold(self):
        grayscale_image = convert_to_grayscale(self.testImg)
        thresh_image = apply_threshold(grayscale_image)
        self.assertEqual(len(thresh_image.shape), 2, "Threshold Image Shape Mismatch")
        self.assertTrue(np.array_equal(np.unique(thresh_image), [0, 255]), "Threshing failed")

    def test_process_image(self):
        processed_image = process_image(self.imgPath)
        self.assertIsNotNone(processed_image, "Failed to Process Image")
        self.assertEqual(set(np.unique(processed_image)), {0,255})

    def test_process_all_images(self):
        #Create a temporary directory that holds test images
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(3):
                cv.imwrite(os.path.join(temp_dir, f"test_image_{i}.png"), self.testImg)
            
            # Process all images in the temporary directory
            processed_images = process_all_images(temp_dir)
            self.assertEqual(len(processed_images), 3)

# Run the tests
if __name__ == "__main__":
    unittest.main()
