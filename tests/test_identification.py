import pytest
from models.identification_model import load_identification_model, identify_objects

def test_identification():
    model = load_identification_model()
    image_path = 'data/input_images/sample.jpg'  # Ensure a sample image is in this path
    labels, scores = identify_objects(image_path, model)
    assert len(labels) > 0
    assert len(scores) > 0
