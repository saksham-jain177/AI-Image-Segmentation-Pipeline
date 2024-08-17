import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Load a pre-trained ResNet model for identification
class IdentificationModel(nn.Module):
    def __init__(self):
        super(IdentificationModel, self).__init__()
        # Load a pre-trained ResNet model and modify the final layer
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Assuming 10 classes for identification

    def forward(self, x):
        return self.model(x)

def identify_object(image_path, model):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Perform identification
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    return torch.argmax(output, dim=1).item()

if __name__ == "__main__":
    # Initialize the model
    identification_model = IdentificationModel()

    # Directory containing the segmented object images
    segmented_image_directory = "E:\saksham-jain-wasserstoff-AiInternTask\data\segmented_objects"

    # Loop through all images in the segmented objects directory
    for filename in os.listdir(segmented_image_directory):
        file_path = os.path.join(segmented_image_directory, filename)
        if os.path.isfile(file_path):
            try:
                identified_class = identify_object(file_path, identification_model)
                print(f"File: {filename}, Identified Class: {identified_class}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
