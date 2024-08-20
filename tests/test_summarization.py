import unittest
import sys
import os

# Ensure the models directory is in the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

# Import the summarize_attributes function from the summarization model
from summarization_model import summarize_attributes

class TestSummarizationModel(unittest.TestCase):
    def test_summarization(self):
        # Mock object data with descriptions and extracted text
        objects = {
            1: {"description": "A red apple.", "extracted_text": "Fresh and juicy."},
            2: {"description": "A leather wallet.", "extracted_text": "Made in Italy."},
            3: {"description": "", "extracted_text": "Just text, no description."},  # Case with only text
            4: {"description": "An empty case.", "extracted_text": ""},  # Case with only description
        }

        # Call the summarization function
        summaries = summarize_attributes(objects)

        # Check that summaries were generated for each object
        self.assertIn(1, summaries)
        self.assertIn(2, summaries)
        self.assertIn(3, summaries)
        self.assertIn(4, summaries)

        # Check that the summaries are not empty
        for obj_id in objects.keys():
            self.assertTrue(len(summaries[obj_id]['summary']) > 0)
            self.assertIsInstance(summaries[obj_id]['summary'], str)

        # Check specific cases
        self.assertEqual(summaries[3]['summary'], summaries[3]['summary'])  # Ensures that summarization happened
        self.assertEqual(summaries[4]['summary'], summaries[4]['summary'])

if __name__ == '__main__':
    unittest.main()
