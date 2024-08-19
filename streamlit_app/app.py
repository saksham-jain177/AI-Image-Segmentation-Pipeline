import sys
import os

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import json
from PIL import Image
from utils.visualization import visualize_segmented_object


SUMMARY_PATH = "E:\saksham-jain-wasserstoff-AiInternTask\data\output\summary.json"
SEGMENTED_OBJECTS_DIR = "E:\saksham-jain-wasserstoff-AiInternTask\data\segmented_objects"

def load_summaries():
    """Load the processed summaries from a JSON file."""
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, 'r') as f:
            return json.load(f)
    else:
        st.error(f"Summary file not found at {SUMMARY_PATH}")
        return {}

def display_segmented_objects():
    """Display segmented objects with their summaries."""
    summaries = load_summaries()
    if not summaries:
        st.warning("No summaries available to display.")
        return

    # Display the segmented objects with their summaries
    for obj_id, obj_data in summaries.items():
        st.subheader(f"Object {obj_id}")
        class_name = obj_data.get("Identified Class", "Unknown")
        extracted_text = obj_data.get("Extracted Text", "No Text")
        
        # Display class name and extracted text
        st.write(f"**Class**: {class_name}")
        st.write(f"**Extracted Text**: {extracted_text}")

        # Display the corresponding segmented object image
        image_name = obj_id.lower().replace(" ", "_") + ".png"
        image_path = os.path.join(SEGMENTED_OBJECTS_DIR, image_name)

        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=f"Segmented Object {obj_id}", use_column_width=True)
        else:
            st.warning(f"Image not found for Object {obj_id}.")

def main():
    """Main function to run the Streamlit app."""
    st.title("Segmented Object Visualization")

    # Display the segmented objects with their summaries
    display_segmented_objects()

if __name__ == "__main__":
    main()
