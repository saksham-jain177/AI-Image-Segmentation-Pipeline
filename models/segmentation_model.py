import torch
from torchvision import models, transforms
from PIL import Image
import os

weights = models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.maskrcnn_resnet50_fpn(weights=weights)
model.eval()

# Transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor()
])

def segment_image(image_path, output_dir):
    # Load and transform the image
    print(f"Loading image from {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Perform segmentation
    print("Performing segmentation...")
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Check if any masks are detected
    if not predictions[0]['masks'].size(0):
        print("No objects detected in the image.")
        return

    # Get the masks
    masks = predictions[0]['masks'].cpu().numpy()
    print(f"Detected {len(masks)} objects in the image.")

    # Save each mask as a separate image
    for i, mask in enumerate(masks):
        mask_image = Image.fromarray((mask[0] * 255).astype('uint8'))
        mask_save_path = os.path.join(output_dir, f"segmented_object_{i+1}.png")
        mask_image.save(mask_save_path)
        print(f"Saved segmented object {i+1} to {mask_save_path}")

if __name__ == "__main__":
    # Example usage
    input_image_path = "E:\saksham-jain-wasserstoff-AiInternTask\data\input_images\sample1.jpg"  # Change this to your actual image path
    output_directory = "E:\saksham-jain-wasserstoff-AiInternTask\data\segmented_objects"
    os.makedirs(output_directory, exist_ok=True)
    segment_image(input_image_path, output_directory)
