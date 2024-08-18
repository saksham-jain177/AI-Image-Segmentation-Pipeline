import json
import os

def clean_data(summary):
    """
    Clean and structure the summarized data.
    Args:
        summary (dict): The raw summary data from the models.
    
    Returns:
        dict: Cleaned and structured summary data.
    """
    cleaned_summary = {}
    
    for key, value in summary.items():
        # Example: Remove objects with no identified class or extracted text
        if value["Identified Class"] != "Unknown" and value["Extracted Text"] != "":
            cleaned_summary[key] = value
    
    return cleaned_summary

def save_cleaned_data(cleaned_summary, output_path="data/output/cleaned_summary.json"):
    """
    Save the cleaned summary data to a JSON file.
    Args:
        cleaned_summary (dict): The cleaned summary data.
        output_path (str): Path to save the cleaned summary.
    """
    with open(output_path, 'w') as f:
        json.dump(cleaned_summary, f, indent=4)
    print(f"Cleaned summary saved to {output_path}")

if __name__ == "__main__":
    summary_path = "data/output/summary.json"
    
    # Check if summary file exists
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Clean the data
        cleaned_summary = clean_data(summary)
        
        # Save the cleaned data
        save_cleaned_data(cleaned_summary)
    else:
        print(f"Summary file not found at {summary_path}")
