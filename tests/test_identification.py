import sys
import os
import torch
import unittest

# Add the models directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from identification_model import IdentificationModel, identify_object

class TestIdentificationModel(unittest.TestCase):

    def setUp(self):
        self.model = IdentificationModel()

    def test_identify_object(self):
        test_image_path = 'E:/saksham-jain-wasserstoff-AiInternTask/data/segmented_objects/segmented_object_1.png'
        identified_class = identify_object(test_image_path, self.model)
        self.assertIsInstance(identified_class, int)

if __name__ == "__main__":
    unittest.main()
