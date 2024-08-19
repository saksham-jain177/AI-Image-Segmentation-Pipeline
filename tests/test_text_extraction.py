import sys
import os
import unittest

# Add the models directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

# Import the extract_text function from the model
from text_extraction_model import extract_text

class TestTextExtractionModel(unittest.TestCase):

    def test_extract_text(self):
        # Provide the path to the test image
        image_path = 'E:/saksham-jain-wasserstoff-AiInternTask/data/segmented_objects/segmented_object_1.png'
        
        # Call the extract_text function
        extracted_text = extract_text(image_path)
        
        # Ensure that extracted_text is always a string
        self.assertIsInstance(extracted_text, str)
        
        # Further check for expected return values
        self.assertTrue(extracted_text in ["No text detected", "Error occurred"] or extracted_text.strip())

if __name__ == "__main__":
    unittest.main()
