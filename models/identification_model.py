import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load a pre-trained ResNet model for identification
class IdentificationModel(nn.Module):
    def __init__(self):
        super(IdentificationModel, self).__init__()
        # Load a pre-trained ResNet model and modify the final layer for 10 classes
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Assuming 10 classes for identification

    def forward(self, x):
        return self.model(x)

def load_model_weights(model, weight_path):
    """ Load the model weights if available. """
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        logging.info(f"Loaded model weights from {weight_path}")
    else:
        logging.warning(f"No weights file found at {weight_path}. Using the model without pre-trained weights.")

def identify_object(image_path, model, device):
    """ Process an image and identify the class. """
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Perform identification
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
        return torch.argmax(output, dim=1).item()
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return None

if __name__ == "__main__":
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize the model and move it to the device
    identification_model = IdentificationModel().to(device)

    # Path to model weights (adjust as needed)
    model_weights_path = "models/identification_model_weights.pth"
    load_model_weights(identification_model, model_weights_path)

    # Directory containing the segmented object images
    segmented_image_directory = r"E:\saksham-jain-wasserstoff-AiInternTask\data\segmented_objects"

    # List to store descriptions
    descriptions = []

    # Loop through all images in the segmented objects directory
    for filename in os.listdir(segmented_image_directory):
        file_path = os.path.join(segmented_image_directory, filename)
        if os.path.isfile(file_path):
            identified_class = identify_object(file_path, identification_model, device)
            if identified_class is not None:
                # Store the identified class as a description
                descriptions.append(f"Object {filename} identified as class {identified_class}")
                logging.info(f"File: {filename}, Identified Class: {identified_class}")
            else:
                logging.error(f"Failed to identify class for {filename}")

    # Print or save the descriptions list for further processing
    logging.info(f"Collected Descriptions: {descriptions}")
