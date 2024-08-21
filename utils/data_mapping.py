import json
import os

def map_data(segmented_objects, descriptions, extracted_texts, summaries, master_image_id, output_file):
    # Add a debug statement to ensure the function is called
    print("map_data() called")
    print(f"Segmented Objects: {segmented_objects}")
    print(f"Descriptions: {descriptions}")
    print(f"Extracted Texts: {extracted_texts}")
    print(f"Summaries: {summaries}")
    print(f"Master Image ID: {master_image_id}")
    print(f"Output File: {output_file}")
    
    # Ensure all lists have the same length
    num_objects = len(segmented_objects)
    if not (num_objects == len(descriptions) == len(extracted_texts) == len(summaries)):
        raise ValueError("Mismatch in the number of segmented objects, descriptions, texts, and summaries")

    data_mapping = {}

    # Map each object with its data
    for i in range(num_objects):
        object_data = {
            "description": descriptions[i],
            "extracted_text": extracted_texts[i],
            "summary": summaries[i]
        }
        data_mapping[f"object_{i+1}"] = object_data
        print(f"Mapped object_{i+1}: {object_data}")  # Debug each object

    # Include master image ID
    data_mapping["master_image_id"] = master_image_id

    # Save the mapping to a JSON file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data_mapping, f, indent=4)
        print(f"Data mapping saved to {output_file}")  # Success message
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")
