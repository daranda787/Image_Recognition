import os
import cv2 as cv
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
from document_conversion.converter import convert_to_cv_image

# Flag to control image display
DISPLAY_IMAGES = False  # Set to True if you want to see each processed image

#  Read & Display an Image
def load_img(path):
    # For all file types, use cv2.imread
    image = cv.imread(path)
    if image is None:
        print(f"Error: Unable to read the file '{path}'. Please verify the file path and format.")
    return image

def display_image(image, title="Image"):
    if DISPLAY_IMAGES:
        plt.figure(figsize=(10, 8))
        if len(image.shape) == 2:  # Grayscale image
            plt.imshow(image, cmap='gray')
        else:  # Color image
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        print(f"Displaying image: {title}")
        print(f"Image shape: {image.shape}")
        print(f"Image data type: {image.dtype}")
        print(f"Image min value: {np.min(image)}, max value: {np.max(image)}")
        plt.show()
    else:
        print(f"Image display disabled. Title: {title}")

#  Convert Image to Grayscale
def convert_to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#  Apply Thresholding
def apply_threshold(image):
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

# Main processing function
def process_image(image_path):
    print(f"Processing file: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return None

    try:
        # Load and convert the image
        image = convert_to_cv_image(image_path)
        print(f"Successfully loaded image: {image_path}")
    except ValueError as e:
        print(str(e))
        return None

    # Display original image
    display_image(image, f"Original Image: {os.path.basename(image_path)}")

    # Convert to grayscale
    gray_image = convert_to_grayscale(image)
    display_image(gray_image, f"Grayscale Image: {os.path.basename(image_path)}")

    # Apply thresholding
    thresholded_image = apply_threshold(gray_image)
    display_image(thresholded_image, f"Thresholded Image: {os.path.basename(image_path)}")

    return thresholded_image

def process_all_images(directory):
    processed_images = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(directory, filename)
            print(f"Processing {filename}...")
            processed_image = process_image(file_path)
            if processed_image is not None:
                processed_images.append((filename, processed_image))
    return processed_images

# Example usage
if __name__ == "__main__":
    image_directory = r"data\CNNPng"  # Updated path to the CNNPng folder
    print(f"Looking for images in: {os.path.abspath(image_directory)}")
    
    files_in_directory = os.listdir(image_directory)
    print(f"Files found in directory: {files_in_directory}")
    
    processed_images = process_all_images(image_directory)
    print(f"Successfully processed {len(processed_images)} images.")
    
    # Here you can add further processing or OCR for all processed images
    for filename, image in processed_images:
        # Add your OCR or other processing steps here
        print(f"Further processing for {filename}")
