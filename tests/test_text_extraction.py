import pytest
from models.text_extraction_model import load_text_extraction_model, extract_text

def test_text_extraction():
    model = load_text_extraction_model()
    image_path = 'data/input_images/sample.jpg'  # Ensure a sample image with text is in this path
    text = extract_text(image_path, model)
    assert len(text) > 0
