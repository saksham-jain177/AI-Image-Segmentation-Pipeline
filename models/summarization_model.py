import json
from identification_model import IdentificationModel, identify_object
from text_extraction_model import extract_text

def generate_summary(identified_objects, extracted_texts):
    summary = {}
    
    for i, obj in enumerate(identified_objects):
        summary[f"Segmented Object {i+1}"] = {
            "Identified Class": obj,
            "Extracted Text": extracted_texts[i]
        }
    
    return summary

def save_summary_to_file(summary, output_path="data/output/summary.json"):
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {output_path}")

if __name__ == "__main__":
    # Initialize the identification model directly without loading weights
    identification_model = IdentificationModel()

    # Actual data from identification and text extraction models
    segmented_objects_dir = "data/segmented_objects/"
    segmented_files = [
        "segmented_object_1.png", 
        "segmented_object_2.png", 
        "segmented_object_3.png",
        "segmented_object_4.png", 
        "segmented_object_5.png", 
        "segmented_object_6.png",
        "segmented_object_7.png", 
        "segmented_object_8.png"
    ]
    
    identified_objects = []
    extracted_texts = []
    
    for segmented_file in segmented_files:
        segmented_path = segmented_objects_dir + segmented_file
        
        # Fetch identified class by passing the model
        identified_class = identify_object(segmented_path, identification_model)  # Pass the model as an argument
        identified_objects.append(identified_class)
        
        # Fetch extracted text
        extracted_text = extract_text(segmented_path)
        extracted_texts.append(extracted_text)

    # Generate summary
    summary = generate_summary(identified_objects, extracted_texts)
    
    # Save summary to file
    save_summary_to_file(summary)
