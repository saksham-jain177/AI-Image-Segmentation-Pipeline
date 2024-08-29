## Project Overview
This project is an AI pipeline for image segmentation, object identification, text extraction, and summarization.

## Installation
1. Clone the repository: 
    `git clone https://github.com/saksham-jain177/AI-Image-Segmentation-Pipeline.git`

2. Install the required dependencies: 
    `pip install -r requirements.txt`


## Project Structure
AiInternTask/

```
│
├── data/
│   ├── input_images/            # Images to be processed
│   ├── segmented_objects/       # Segmented objects from input images
│   └── output/                  # Final output including summaries
│
├── models/
│   ├── segmentation_model.py    # Segmentation model for image processing
│   ├── identification_model.py  # Model for object identification
│   ├── text_extraction_model.py # Model for extracting text from objects
│   └── summarization_model.py   # Model for summarizing the findings
│
├── utils/
│   ├── preprocessing.py         # Data preprocessing utilities
│   ├── postprocessing.py        # Post-processing utilities
│   ├── data_mapping.py          # Data mapping functions
│   └── visualization.py         # Functions for visualizing results
│
├── streamlit_app/
│   ├── app.py                   # Streamlit app for running the pipeline
│   └── components/              # UI components
│
├── tests/                       # Unit tests for various components
│   ├── test_segmentation.py
│   ├── test_identification.py
│   ├── test_text_extraction.py
│   └── test_summarization.py
│
├── README.md
└── requirements.txt
```

## Usage
1. Run the pipeline:
    `streamlit run streamlit_app/app.py`
This will launch a web interface where you can upload images, segment them, extract text, and view summaries.

2. Input Data:
Place your input images in the data/input_images/ directory. These images will be processed by the pipeline.

3. Output Data:
The segmented objects, extracted text, and final summaries will be saved in the data/output/ directory.

4. Clearing Previous Data:
Before running the pipeline, ensure that the segmented_objects/ and output/ directories are cleared if you want a fresh run.

## Testing
Run unit tests to ensure that each module is working as expected:
    `pytest tests/`
