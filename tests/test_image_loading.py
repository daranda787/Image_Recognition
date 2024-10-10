import unittest
import cv2 as cv
import os

class TestImageLoading(unittest.TestCase):
    def setUp(self):
        #Set up image path
        self.imgPath = 'C:/Users/Darien/Downloads/RefImg.png'

    def test_image_loading(self):
        self.assertTrue(os.path.exists(self.imgPath), "Image File Not Found")

        image = cv.imread(self.imgPath)

        self.assertIsNotNone(image, "Failed to Load Image")

    def test_image_display(self):
        try:
            import matplotlib.pyplot as plt

            image = cv.imread(self.imgPath)
            plt.imshow(cv.cvtColor(image,cv.COLOR_BRG2RGB))
            plt.axis('off')
            plt.show()
        except Exception as e:
            self.fail(f"Displaying the image failed with error: {e}")

if __name__ == "__main__":
    unittest.main()