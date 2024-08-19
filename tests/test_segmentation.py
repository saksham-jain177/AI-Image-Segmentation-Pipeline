import sys
import os
import torch
import unittest

# Add the models directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from segmentation_model import segment_image

class TestSegmentationModel(unittest.TestCase):

    def test_segment_image(self):
        input_image_path = 'E:/saksham-jain-wasserstoff-AiInternTask/data/input_images/sample1.jpg'
        output_dir = 'E:/saksham-jain-wasserstoff-AiInternTask/data/segmented_objects'
        os.makedirs(output_dir, exist_ok=True)

        segment_image(input_image_path, output_dir)

        segmented_files = os.listdir(output_dir)
        self.assertGreater(len(segmented_files), 0)

if __name__ == "__main__":
    unittest.main()
