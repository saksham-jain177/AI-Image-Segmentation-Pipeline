import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image_path):
    try:
        # Load the image
        img = Image.open(image_path)

        # Apply preprocessing (convert to grayscale, enhance contrast, etc.)
        img = img.convert('L')  # Convert to grayscale
        img = img.filter(ImageFilter.SHARPEN)  # Sharpen image
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)  # Enhance contrast

        # Extract text using Tesseract
        text = pytesseract.image_to_string(img)
        return text if text.strip() else "No text detected"
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
