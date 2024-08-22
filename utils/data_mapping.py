from models.identification_model import IdentificationModel, identify_object
from models.text_extraction_model import extract_text
from transformers import pipeline
import torch
import json
import os

# Path configurations
SEGMENTED_OBJECTS_DIR = "E:/saksham-jain-wasserstoff-AiInternTask/data/segmented_objects"
SUMMARY_OUTPUT_PATH = "E:/saksham-jain-wasserstoff-AiInternTask/data/output/summary.json"

# Initialize models
identification_model = IdentificationModel()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
identification_model = identification_model.to(device)

# Initialize summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

def generate_summary(identified_class, extracted_text):
    """ Generate a summary for each identified object and its extracted text. """
    input_text = f"Object: {identified_class}, Text: {extracted_text}"
    try:
        summary = summarizer(input_text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing: {e}")
        summary = "Error in summary"
    return summary

if __name__ == "__main__":
    # Prepare the data for mapping
    data_mappings = []

    # Loop through each segmented object
    for filename in os.listdir(SEGMENTED_OBJECTS_DIR):
        file_path = os.path.join(SEGMENTED_OBJECTS_DIR, filename)
        
        if os.path.isfile(file_path):
            # Identify the object class
            identified_class = identify_object(file_path, identification_model, device)

            # Extract the text from the segmented object
            extracted_text = extract_text(file_path)

            # Generate a summary based on the identified class and extracted text
            summary = generate_summary(identified_class, extracted_text)

            # Create the mapping for this segmented object
            data_mappings.append({
                "segmented_object": filename,
                "identified_class": identified_class,
                "extracted_text": extracted_text,
                "summary": summary
            })

    # Save the mapping to the summary file
    with open(SUMMARY_OUTPUT_PATH, 'w') as summary_file:
        json.dump(data_mappings, summary_file, indent=4)

    print(f"Data mapping completed and saved to {SUMMARY_OUTPUT_PATH}")
