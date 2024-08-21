import streamlit as st
from PIL import Image
import os
import torch
import json
import sys
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import models and utilities
from models.segmentation_model import segment_image
from models.identification_model import IdentificationModel, identify_object
from models.text_extraction_model import extract_text
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from utils.data_mapping import generate_summary

# Directory paths
segmented_objects_dir = "E:\saksham-jain-wasserstoff-AiInternTask\data\segmented_objects"
output_dir = "E:\saksham-jain-wasserstoff-AiInternTask\data\output"

# Initialize models
identification_model = IdentificationModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
identification_model.to(device)

# Streamlit app setup
st.title("AI Image Processing Pipeline")

# File uploader for input image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Save the uploaded image temporarily to the input_images directory
    image_path = os.path.join("E:\saksham-jain-wasserstoff-AiInternTask\data\input_images", uploaded_image.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Load and display the image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 1: Image Segmentation
    st.write("## Step 1: Image Segmentation")
    
    # Segment the image and save the results to the segmented_objects directory
    segment_image(image_path, segmented_objects_dir)

    # Load segmented objects from the directory and display them
    segmented_objects = [Image.open(os.path.join(segmented_objects_dir, f)) for f in os.listdir(segmented_objects_dir)]

    for i, obj in enumerate(segmented_objects):
        st.image(obj, caption=f"Segmented Object {i+1}", use_column_width=True)

    st.write(f"Segmented {len(segmented_objects)} objects.")

    # Step 2: Object Extraction and Storage
    st.write("## Step 2: Object Extraction and Storage")
    master_image_id = uploaded_image.name
    object_metadata = {}

    for i, obj in enumerate(segmented_objects):
        obj_id = f"object_{i+1}"
        obj_path = os.path.join(segmented_objects_dir, f"segmented_object_{i+1}.png")

        # Assign a unique ID and save metadata
        object_metadata[obj_id] = {"image_path": obj_path, "master_id": master_image_id}
        st.write(f"Object {i+1}: Saved with ID {obj_id} and Master ID {master_image_id}")

    # Step 3: Object Identification
    st.write("## Step 3: Object Identification")
    identified_classes = []

    for i, segmented_object in enumerate(segmented_objects):
        obj_id = f"object_{i+1}"
        obj_path = os.path.join(segmented_objects_dir, f"segmented_object_{i+1}.png")

        # Object identification
        identified_class = identify_object(obj_path, identification_model, device)
        identified_classes.append(identified_class)
        object_metadata[obj_id]["identified_class"] = identified_class

        st.write(f"Object {i+1}: Identified Class: {identified_class}")

    # Step 4: Text/Data Extraction from Objects
    st.write("## Step 4: Text/Data Extraction from Objects")
    extracted_texts = []

    for i, segmented_object in enumerate(segmented_objects):
        obj_id = f"object_{i+1}"
        obj_path = os.path.join(segmented_objects_dir, f"segmented_object_{i+1}.png")

        # Text extraction
        extracted_text = extract_text(obj_path)
        extracted_texts.append(extracted_text)
        object_metadata[obj_id]["extracted_text"] = extracted_text

        st.write(f"Object {i+1}: Extracted Text: {extracted_text}")

    # Step 5: Summarize Object Attributes
    st.write("## Step 5: Summarize Object Attributes")
    summaries = []

    for i, segmented_object in enumerate(segmented_objects):
        obj_id = f"object_{i+1}"

        # Generate summary using the `generate_summary` function from `data_mapping.py`
        description = object_metadata[obj_id].get("identified_class", "")
        extracted_text = object_metadata[obj_id].get("extracted_text", "")
        summary = generate_summary(description, extracted_text)

        summaries.append(summary)
        object_metadata[obj_id]["summary"] = summary

        st.write(f"Object {i+1} Summary: {summary}")

    # Step 6: Save Data and Summaries
    st.write("## Step 6: Save Data and Summaries")
    output_file = os.path.join(output_dir, "summary.json")

    # Save object metadata and summaries to JSON
    with open(output_file, 'w') as f:
        json.dump(object_metadata, f, indent=4)
    st.write(f"Data and summaries saved to {output_file}")

    # Step 7: Results Download
    st.write("## Step 7: Download Summary")
    with open(output_file, "r") as f:
        summary_json = f.read()

    st.download_button(label="Download Summary", data=summary_json, file_name="summary.json")
