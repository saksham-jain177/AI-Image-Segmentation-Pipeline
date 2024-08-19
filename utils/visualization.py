import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import math

def visualize_segmented_object(image_path, title=None):
    """
    Return the image and title for visualization in a grid layout.
    
    Args:
        image_path (str): The path to the segmented object image.
        title (str, optional): The title to display above the image (e.g., class name, extracted text).
    
    Returns:
        tuple: (image, title)
    """
    image = Image.open(image_path)
    return image, title

def visualize_summary(summary_path="data/output/summary.json", segmented_images_dir="data/segmented_objects"):
    """
    Visualize all segmented object images and their summaries in a grid format.
    
    Args:
        summary_path (str): The path to the summary JSON file.
        segmented_images_dir (str): The directory containing segmented object images.
    """
    # Load and process the summaries
    with open(summary_path, 'r') as f:
        processed_summaries = json.load(f)
    
    images_and_titles = []
    
    for obj_key, obj_data in processed_summaries.items():
        image_name = obj_key.lower().replace(" ", "_") + ".png"
        image_path = os.path.join(segmented_images_dir, image_name)
        
        if os.path.exists(image_path):
            title = f"Class: {obj_data['Identified Class']}, Text: {obj_data.get('Extracted Text', 'No Text')}"
            images_and_titles.append(visualize_segmented_object(image_path, title=title))
        else:
            print(f"Image not found for {obj_key} at {image_path}")
    
    # Determine grid size for displaying all images
    num_images = len(images_and_titles)
    grid_size = math.ceil(math.sqrt(num_images))  # Create a square grid
    
    # Plot all images in a grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    for idx, (image, title) in enumerate(images_and_titles):
        row, col = divmod(idx, grid_size)
        axes[row, col].imshow(image)
        axes[row, col].set_title(title)
        axes[row, col].axis("off")
    
    # Remove any unused subplots
    for idx in range(len(images_and_titles), grid_size * grid_size):
        row, col = divmod(idx, grid_size)
        axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    visualize_summary()
