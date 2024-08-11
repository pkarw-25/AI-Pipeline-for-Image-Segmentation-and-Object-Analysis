from PIL import Image
import os
import numpy as np
def save_segmented_objects(image, masks, labels, output_dir, master_id):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    object_info = []
    
    for idx, mask in enumerate(masks):
        object_id = f"{master_id}_obj_{idx}"
        mask = mask[0]
        
        segmented_image = Image.fromarray((mask * 255).astype(np.uint8))
        object_path = os.path.join(output_dir, f"{object_id}.png")
        segmented_image.save(object_path)
        
        object_info.append({
            "object_id": object_id,
            "master_id": master_id,
            "path": object_path
        })
    
    return object_info
