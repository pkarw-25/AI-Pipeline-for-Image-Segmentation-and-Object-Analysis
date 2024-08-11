import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_identification(image_path, labels, scores, output_path):
    image = plt.imread(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for i in range(len(labels)):
        score = scores[i]
        if score > 0.5:  # Confidence threshold
            rect = patches.Rectangle((0, 0), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()
