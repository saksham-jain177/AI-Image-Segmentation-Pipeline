import unittest
import sys
import os

# Ensure the models directory is in the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

# Import the summarize_attributes function from the summarization model
from summarization_model import summarize_attributes

class TestSummarizationModel(unittest.TestCase):
    def test_summarization(self):
        # Mock object data
        objects = {
            1: {"description": "A red apple.", "extracted_text": "Fresh and juicy."},
            2: {"description": "A leather wallet.", "extracted_text": "Made in Italy."}
        }

        # Call the summarization function
        summaries = summarize_attributes(objects)

        # Check that summaries were generated
        self.assertIn(1, summaries)
        self.assertIn(2, summaries)
        self.assertTrue(len(summaries[1]['summary']) > 0)
        self.assertTrue(len(summaries[2]['summary']) > 0)

if __name__ == '__main__':
    unittest.main()
