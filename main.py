import pytesseract
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt

# Flag to control image display
DISPLAY_IMAGES = True  # Set to False to disable image display

#  Read & Display an Image
def load_img(path):
    image = cv.imread(path)
    
    if image is None:
        print("Error: Unable to read the image. Please verify the file path.")
        return None
    
    return image

def display_image(image, title="Image"):
    if DISPLAY_IMAGES:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        print(f"Image display disabled. Title: {title}")

#  Convert Image to Grayscale
def convert_to_grayscale(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if DISPLAY_IMAGES:
        display_image(gray_image, "Grayscale Image")
    return gray_image

#  Apply Thresholding
def apply_th(image):
    thresholded = cv.threshold(image, 127, 255, cv.THRESH_BINARY)[1]
    if DISPLAY_IMAGES:
        display_image(thresholded, "Thresholded Image")
    return thresholded

#Resize Image (Optional)
def resize_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    if DISPLAY_IMAGES:
        display_image(resized, f"Resized Image ({scale_percent}%)")
    return resized

# Main processing function
def process_image(image_path):
    # Load the image
    image = load_img(image_path)
    if image is None:
        return

    # Display original image
    display_image(image, "Original Image")

    # Convert to grayscale
    gray_image = convert_to_grayscale(image)

    # Apply thresholding
    thresholded_image = apply_th(gray_image)

    # Optionally resize
    # resized_image = resize_image(thresholded_image, 75)

    return thresholded_image

# Example usage
if __name__ == "__main__":
    image_path = r"data\cnn.pdf"
    processed_image = process_image(image_path)
    # Further processing or OCR can be done here
