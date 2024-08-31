import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import os

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Apply dilation to connect text components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.dilate(gray, kernel, iterations=1)
    
    # Apply erosion to remove noise
    gray = cv2.erode(gray, kernel, iterations=1)
    
    return Image.fromarray(gray)

def extract_text(image_path):
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Preprocess the image
        img = preprocess_image(img)
        
        # Extract text using Tesseract with custom configuration
        custom_config = r'--oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        text = pytesseract.image_to_string(img, config=custom_config)
        
        return text.strip() if text.strip() else "No text detected"
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return "Error extracting text"

if __name__ == "__main__":
    # Directory containing the segmented objects
    segmented_objects_dir = 'E:/saksham-jain-wasserstoff-AiInternTask/data/segmented_objects'

    # Iterate over the segmented object images
    for image_file in os.listdir(segmented_objects_dir):
        image_path = os.path.join(segmented_objects_dir, image_file)
        
        # Ensure it's a file before processing
        if os.path.isfile(image_path):
            text = extract_text(image_path)
            
            # Print the extracted text for each file
            print(f"File: {image_file}, Extracted Text: {text}")