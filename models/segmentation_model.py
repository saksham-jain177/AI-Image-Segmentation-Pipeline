import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

# Load pre-trained Mask R-CNN model
weights = models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.maskrcnn_resnet50_fpn(weights=weights)
model.eval()

# Transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor()
])

def segment_image(image_path, output_dir, min_mask_area=500,resize_to=(800, 800)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(resize_to)
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    masks = predictions[0]['masks'].cpu().numpy()
    filtered_masks = [mask for mask in masks if np.sum(mask) > min_mask_area]

    if not filtered_masks:
        print(f"No sufficiently large objects detected in {os.path.basename(image_path)}.")
        return

    print(f"Detected {len(filtered_masks)} objects in {os.path.basename(image_path)}.")

    for i, mask in enumerate(filtered_masks):
        mask_image = Image.fromarray((mask[0] * 255).astype('uint8'))
        mask_save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_segmented_object_{i+1}.png")
        mask_image.save(mask_save_path)
        print(f"Saved {mask_save_path}")

if __name__ == "__main__":
    input_images_dir = "E:/saksham-jain-wasserstoff-AiInternTask/data/input_images"
    output_directory = "E:/saksham-jain-wasserstoff-AiInternTask/data/segmented_objects"
    os.makedirs(output_directory, exist_ok=True)

    for image_file in os.listdir(input_images_dir):
        image_path = os.path.join(input_images_dir, image_file)
        if os.path.isfile(image_path):
            segment_image(image_path, output_directory)
