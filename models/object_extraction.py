from PIL import Image
import numpy as np
import os

def extract_objects(image, masks, labels, base_dir='extracted_objects', master_id=None):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    results = []
    for i, (mask, label) in enumerate(zip(masks, labels)):
        mask = mask[0] > 0.5
        object_image = Image.fromarray((np.array(image) * mask.numpy()).astype(np.uint8))
        obj_id = f'{master_id}_{i}'
        object_image.save(f'{base_dir}/{obj_id}.png')
        results.append({'object_id': obj_id, 'master_id': master_id, 'label': label.item()})
    return results
