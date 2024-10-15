import cv2
import numpy as np

def convert_to_cv_image(file_path):
    """
    Convert various file types to a format compatible with OpenCV.
    Currently supports: PNG, JPEG, BMP, TIFF
    """
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Unable to read file: {file_path}")
    return image