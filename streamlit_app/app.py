import streamlit as st
import os
import json
from PIL import Image, ImageDraw

def load_data_mapping(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def display_image_with_annotations(image_path, data_mapping):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Annotate each object in the image (coordinates can be dummy for now)
    for i, (object_id, object_data) in enumerate(data_mapping.items()):
        if object_id != "master_image_id":
            description = object_data["description"]
            draw.text((10, 10 + i * 20), f"{object_id}: {description}", fill="red")

    st.image(image, caption=f"Image with Annotations: {os.path.basename(image_path)}")

def display_data_table(data_mapping):
    table_data = []
    for object_id, object_data in data_mapping.items():
        if object_id != "master_image_id":
            table_data.append([object_id, object_data["description"], object_data["extracted_text"], object_data["summary"]])

    st.table(table_data)

def main():
    st.title("Object Data and Annotations")

    # Folder paths
    input_images_dir = "E:/saksham-jain-wasserstoff-AiInternTask/data/input_images"
    output_dir = "E:/saksham-jain-wasserstoff-AiInternTask/data/output"
    
    # Iterate over all images in the input_images directory
    for image_file in os.listdir(input_images_dir):
        image_path = os.path.join(input_images_dir, image_file)
        
        if os.path.isfile(image_path):
            # Set the master image ID as the image filename without extension
            master_image_id = os.path.splitext(image_file)[0]
            
            # JSON file corresponding to the current image
            json_file = os.path.join(output_dir, f"{master_image_id}_summary.json")
            
            if os.path.exists(json_file):
                # Load the mapped data for this image
                data_mapping = load_data_mapping(json_file)

                # Display the original image with annotations
                display_image_with_annotations(image_path, data_mapping)

                # Display the data table
                st.subheader(f"Data Table for {image_file}")
                display_data_table(data_mapping)
            else:
                st.write(f"No data mapping found for {image_file}")

if __name__ == "__main__":
    main()
