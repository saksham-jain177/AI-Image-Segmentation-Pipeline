import json

# Example mapping: Class IDs to Human-readable Class Names
CLASS_MAPPING = {
    0: "Person",
    1: "Car",
    2: "Tree",
    3: "Dog",
    4: "Building",
    5: "Bicycle",
    6: "Laptop",
    7: "Book",
    8: "Chair",
    9: "Table"
}

def map_identified_classes(summary):
    """
    Map identified class IDs to human-readable class names.
    
    Args:
        summary (dict): The cleaned summary data with class IDs.
    
    Returns:
        dict: The summary data with mapped class names.
    """
    mapped_summary = {}
    
    for key, value in summary.items():
        class_id = value.get("Identified Class")
        if class_id in CLASS_MAPPING:
            value["Identified Class"] = CLASS_MAPPING[class_id]
        else:
            value["Identified Class"] = "Unknown"
        
        mapped_summary[key] = value
    
    return mapped_summary

def save_mapped_data(mapped_summary, output_path="data/output/mapped_summary.json"):
    """
    Save the mapped summary data to a JSON file.
    
    Args:
        mapped_summary (dict): The summary data with mapped class names.
        output_path (str): Path to save the mapped summary.
    """
    with open(output_path, 'w') as f:
        json.dump(mapped_summary, f, indent=4)
    print(f"Mapped summary saved to {output_path}")

if __name__ == "__main__":
    # Example usage: Load cleaned summary, map class IDs to class names, and save the mapped version
    cleaned_summary_path = "data/output/cleaned_summary.json"
    
    # Check if cleaned summary file exists
    try:
        with open(cleaned_summary_path, 'r') as f:
            cleaned_summary = json.load(f)
        
        # Map the class IDs to class names
        mapped_summary = map_identified_classes(cleaned_summary)
        
        # Save the mapped summary data
        save_mapped_data(mapped_summary)
    
    except FileNotFoundError:
        print(f"Cleaned summary file not found at {cleaned_summary_path}")
