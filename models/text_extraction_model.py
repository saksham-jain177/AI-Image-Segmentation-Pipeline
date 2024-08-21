import pytesseract
from PIL import Image
import cv2
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image_path):
    try:
        # Load the image using OpenCV in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply adaptive thresholding to improve text detection
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        # Convert back to PIL Image for Tesseract OCR
        pil_img = Image.fromarray(img)

        # Extract text using Tesseract
        text = pytesseract.image_to_string(pil_img, config='--psm 6')

        # Return extracted text or "No text detected" if empty
        return text if text.strip() else "No text detected"
    except Exception as e:
        logging.error(f"Error extracting text from {image_path}: {e}")
        return "Error extracting text"

if __name__ == "__main__":
    # Directory containing the segmented objects
    segmented_objects_dir = 'E:/saksham-jain-wasserstoff-AiInternTask/data/segmented_objects'

    # List to store extracted texts
    extracted_texts = []

    # Loop through all images in the segmented objects directory
    for image_file in os.listdir(segmented_objects_dir):
        image_path = os.path.join(segmented_objects_dir, image_file)
        
        # Ensure it's a file before processing
        if os.path.isfile(image_path):
            text = extract_text(image_path)
            
            # Store the extracted text
            extracted_texts.append(text)
            
            # Log the extracted text for each file
            logging.info(f"File: {image_file}, Extracted Text: {text}")

    # Print or save the extracted_texts list for further processing
    logging.info(f"Collected Extracted Texts: {extracted_texts}")
