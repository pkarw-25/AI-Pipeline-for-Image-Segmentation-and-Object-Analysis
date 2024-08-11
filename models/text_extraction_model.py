import easyocr

def load_text_extraction_model():
    reader = easyocr.Reader(['en'])
    return reader

def extract_text(image_path, model):
    results = model.readtext(image_path)
    text = " ".join([res[1] for res in results])
    return text
