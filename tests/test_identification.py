import sys
import os
import torch
import unittest

# Add the models directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from identification_model import IdentificationModel, identify_object, load_model_weights

class TestIdentificationModel(unittest.TestCase):

    def setUp(self):
        # Initialize the model and move it to the appropriate device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = IdentificationModel().to(self.device)

        # Load pre-trained weights if available
        model_weights_path = "E:/saksham-jain-wasserstoff-AiInternTask/models/identification_model_weights.pth"
        load_model_weights(self.model, model_weights_path)

    def test_identify_object(self):
        # Ensure that the test image exists
        test_image_path = 'E:/saksham-jain-wasserstoff-AiInternTask/data/segmented_objects/segmented_object_1.png'
        self.assertTrue(os.path.exists(test_image_path), "Test image does not exist")

        # Identify the object and assert the identified class is an integer
        identified_class = identify_object(test_image_path, self.model, self.device)
        self.assertIsInstance(identified_class, int)

if __name__ == "__main__":
    unittest.main()
