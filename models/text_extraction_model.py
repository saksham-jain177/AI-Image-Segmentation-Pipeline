import pytesseract
from PIL import Image
import os
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text(image_path):
    try:
        # Open the image and attempt to extract text
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        
        # Check if extracted text is empty or just whitespace
        if not extracted_text.strip():
            return "No text detected"
        
        return extracted_text

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return "Error occurred"


# Directory containing the segmented objects
segmented_objects_dir = 'E:\saksham-jain-wasserstoff-AiInternTask\data\segmented_objects'

# Iterate over the segmented object images
for image_file in os.listdir(segmented_objects_dir):
    image_path = os.path.join(segmented_objects_dir, image_file)
    text = extract_text(image_path)
    
    if text:
        print(f"File: {image_file}, Extracted Text: {text}")
