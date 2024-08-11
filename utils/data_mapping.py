def map_segmented_to_labels(segmented_image):
    labels = {
        0: 'background',
        1: 'person',
        2: 'car',
        # Add more labels as per your segmentation classes
    }
    
    label_image = [[labels[pixel] for pixel in row] for row in segmented_image]
    return label_image
