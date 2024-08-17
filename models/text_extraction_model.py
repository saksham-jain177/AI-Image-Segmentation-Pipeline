import pytesseract
from PIL import Image
import os
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text(image_path):
    try:
        # Try to open the image
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        
        # Check if any text was found
        if text.strip():  # Check if the text is not empty after stripping whitespace
            return text
        else:
            print(f"No text found in {os.path.basename(image_path)}")
            return None
    except Exception as e:
        # Handle exceptions such as issues with Tesseract or file reading
        print(f"Error extracting text from {image_path}: {e}")
        return None

# Directory containing the segmented objects
segmented_objects_dir = 'E:\saksham-jain-wasserstoff-AiInternTask\data\segmented_objects'

# Iterate over the segmented object images
for image_file in os.listdir(segmented_objects_dir):
    image_path = os.path.join(segmented_objects_dir, image_file)
    text = extract_text(image_path)
    
    if text:
        print(f"File: {image_file}, Extracted Text: {text}")
