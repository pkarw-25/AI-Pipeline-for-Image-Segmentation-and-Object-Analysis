import streamlit as st
from PIL import Image
import os
from models.segmentation_model import load_model as load_segmentation_model, segment_image, visualize_segmentation
from models.identification_model import load_identification_model, identify_objects
from models.text_extraction_model import load_text_extraction_model, extract_text
from utils.preprocessing import save_segmented_objects

# Set up directories
input_dir = 'data/input_images/'
segmented_dir = 'data/segmented_objects/'
output_dir = 'data/output/'

# Load models
segmentation_model = load_segmentation_model()
identification_model = load_identification_model()
text_extraction_model = load_text_extraction_model()

# Streamlit app title
st.title('Image Processing Pipeline')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_path = os.path.join(input_dir, uploaded_file.name)
    image = Image.open(uploaded_file).convert("RGB")
    image.save(image_path)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Segment image
    st.write("Segmenting image...")
    image, masks, labels = segment_image(image_path, segmentation_model)

    # Visualize segmentation
    st.write("Visualizing segmented objects...")
    segmented_output_path = os.path.join(output_dir, "segmented_output.png")
    visualize_segmentation(image, masks, segmented_output_path)
    st.image(segmented_output_path, caption='Segmented Image', use_column_width=True)

    # Extract objects
    master_id = os.path.splitext(uploaded_file.name)[0]
    st.write("Extracting objects...")
    extracted_objects = save_segmented_objects(image, masks, labels, segmented_dir, master_id)
    st.write(f"Extracted {len(extracted_objects)} objects.")

    # Identify objects
    st.write("Identifying objects...")
    for obj in extracted_objects:
        obj_path = obj["path"]
        labels, scores = identify_objects(obj_path, identification_model)
        identification_output_path = os.path.join(output_dir, f"{obj['object_id']}_identified.png")
        st.write(f"Object ID: {obj['object_id']}, Labels: {labels}, Scores: {scores}")
        
    # Extract text from objects
    st.write("Extracting text from objects...")
    for obj in extracted_objects:
        obj_path = obj["path"]
        text = extract_text(obj_path, text_extraction_model)
        st.write(f"Object ID: {obj['object_id']}, Extracted Text: {text}")

st.write("Done.")
