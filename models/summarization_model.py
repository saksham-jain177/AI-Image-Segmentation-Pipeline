import json
from identification_model import IdentificationModel, identify_object  # Adjust paths as necessary
from text_extraction_model import extract_text  # Adjust paths as necessary
from transformers import pipeline
import torch
import os

def generate_summary(identified_objects, extracted_texts):
    """
    Generate a summary by combining identified class and extracted text for each segmented object.
    """
    summary = {}
    
    for i, obj in enumerate(identified_objects):
        summary[f"Segmented Object {i+1}"] = {
            "Identified Class": obj,
            "Extracted Text": extracted_texts[i]
        }
    
    return summary

def save_summary_to_file(summary, output_path="data/output/summary.json"):
    """
    Save the generated summary to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {output_path}")

def summarize_attributes(identified_objects, extracted_texts):
    """
    Use a pre-trained summarization model to summarize the attributes of the identified objects and extracted texts.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summarized_data = []

    for i in range(len(identified_objects)):
        input_text = f"Identified Class: {identified_objects[i]}.\nExtracted Text: {extracted_texts[i]}"
        
        # Generate summary using the summarization pipeline
        summary_text = summarizer(input_text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        
        summarized_data.append({
            "Segmented Object": f"Segmented Object {i+1}",
            "Summary": summary_text,
            "Identified Class": identified_objects[i],
            "Extracted Text": extracted_texts[i]
        })

    return summarized_data

if __name__ == "__main__":
    # Initialize the identification model directly
    identification_model = IdentificationModel()
    
    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directory containing segmented objects
    segmented_objects_dir = "E:/saksham-jain-wasserstoff-AiInternTask/data/segmented_objects"
    
    # List to store identified objects and extracted texts
    identified_objects = []
    extracted_texts = []
    
    # Loop through all segmented object images
    for filename in os.listdir(segmented_objects_dir):
        file_path = os.path.join(segmented_objects_dir, filename)
        
        if os.path.isfile(file_path):
            # Identify object class
            identified_class = identify_object(file_path, identification_model, device)
            identified_objects.append(identified_class)
            
            # Extract text from the image
            extracted_text = extract_text(file_path)
            extracted_texts.append(extracted_text)

    # Generate basic summary with identified objects and extracted texts
    summary = generate_summary(identified_objects, extracted_texts)
    
    # Save basic summary to JSON
    save_summary_to_file(summary)
    
    # Generate detailed summarized data
    summarized_data = summarize_attributes(identified_objects, extracted_texts)
    
    # Save the detailed summaries in a separate file
    save_summary_to_file(summarized_data, output_path="data/output/detailed_summary.json")
