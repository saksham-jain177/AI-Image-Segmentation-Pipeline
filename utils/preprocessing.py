import os
from PIL import Image
import torchvision.transforms as transforms
import torch

def preprocess_image(image_path):
    """
    Preprocesses the input image by resizing and normalizing it.

    Args:
        image_path (str): Path to the input image.

    Returns:
        preprocessed_image (torch.Tensor): The preprocessed image tensor.
    """
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Define the transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ImageNet standards
        ])
        
        # Apply the transformations
        preprocessed_image = transform(image)
        return preprocessed_image
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def preprocess_images_in_directory(input_directory):
    """
    Preprocess all images in the specified input directory.
    
    Args:
        input_directory (str): Directory containing the raw images.
        
    Returns:
        preprocessed_images (list): List of preprocessed image tensors.
    """
    preprocessed_images = []
    
    # Loop through all images in the directory
    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)
        if os.path.isfile(file_path):
            preprocessed_image = preprocess_image(file_path)
            if preprocessed_image is not None:
                preprocessed_images.append(preprocessed_image)
    
    return preprocessed_images

if __name__ == "__main__":
    # Example usage based on the existing file structure
    input_dir = "E:/saksham-jain-wasserstoff-AiInternTask/data/input_images"
    
    # Preprocess all images in the input directory
    preprocessed_images = preprocess_images_in_directory(input_dir)
    
    print(f"Total preprocessed images: {len(preprocessed_images)}")
