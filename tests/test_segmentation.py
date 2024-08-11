import pytest
from models.segmentation_model import load_model, segment_image

def test_segmentation():
    model = load_model()
    image_path = 'data/input_images/sample.jpg'  # Ensure a sample image is in this path
    image, masks, labels = segment_image(image_path, model)
    assert len(masks) > 0
    assert len(labels) > 0
