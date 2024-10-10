import pytesseract
import cv2
from PIL import Image
import matplotlib.pyplot as plt

#  Read & Display an Image
def load_std(path):
    imgPath = "data/cnn.pdf"
    image = cv2.imread(imgPath)
    
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()



#  Convert Image to Grayscale


#  Apply Thresholding


#Resize Image (Optional)