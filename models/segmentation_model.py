import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from torchvision.transforms import functional as F
import cv2

# Load pre-trained Mask R-CNN model
weights = models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.maskrcnn_resnet50_fpn(weights=weights)
model.eval()

# Transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor()
])

def segment_image(image_path, output_dir, min_mask_area=500, resize_to=(400, 400)):
    # Open and resize the image
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size
    image = original_image.resize(resize_to)
    
    # Prepare image for model
    image_tensor = F.to_tensor(image).unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Process masks
    masks = predictions[0]['masks'].cpu().numpy()
    filtered_masks = [mask for mask in masks if np.sum(mask) > min_mask_area]
    
    if not filtered_masks:
        print(f"No sufficiently large objects detected in {os.path.basename(image_path)}.")
        return
    
    print(f"Detected {len(filtered_masks)} objects in {os.path.basename(image_path)}.")
    
    # Convert image to numpy array
    image_np = np.array(image)
    
    for i, mask in enumerate(filtered_masks):
        # Threshold the mask
        binary_mask = (mask[0] > 0.5).astype(np.uint8) * 255
        
        # Create RGBA image: RGB from original, A from binary mask
        rgba = np.dstack((image_np, binary_mask))
        
        # Convert to PIL Image and resize back to original size
        segmented_image = Image.fromarray(rgba.astype('uint8'), 'RGBA').resize(original_size)
        
        # Save the segmented image
        mask_save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_segmented_object_{i+1}.png")
        segmented_image.save(mask_save_path)
        print(f"Saved {mask_save_path}")
        
if __name__ == "__main__":
    input_images_dir = "E:/saksham-jain-wasserstoff-AiInternTask/data/input_images"
    output_directory = "E:/saksham-jain-wasserstoff-AiInternTask/data/segmented_objects"
    os.makedirs(output_directory, exist_ok=True)

    for image_file in os.listdir(input_images_dir):
        image_path = os.path.join(input_images_dir, image_file)
        if os.path.isfile(image_path):
            segment_image(image_path, output_directory)
