import os
import numpy as np
from PIL import Image
import submission

Image.MAX_IMAGE_PIXELS = None

def check_image_and_prediction_size(image_path, prediction_csv_path):
    # Load the PPM image to get its dimensions
    image = Image.open(image_path)
    image_width, image_height = image.size  # (width, height)
    
    # Load the CSV prediction
    prediction = np.loadtxt(prediction_csv_path, delimiter=',')
    prediction_height, prediction_width = prediction.shape

    # Check if the dimensions match
    if image_width == prediction_width and image_height == prediction_height:
        print(f"Prediction for {image_path} matches the input image size: ({image_width}, {image_height})")
    else:
        print(f"Size mismatch for {image_path}:")
        print(f"Image size: ({image_width}, {image_height}), Prediction size: ({prediction_width}, {prediction_height})")

# Example usage:
image_path = '/mounts/Datasets3/2024-2025-ChallengePlankton/test/rg20090520_scan.png.ppm'  # Path to the PPM image
prediction_csv_path = 'predictions/prediction_0.csv'  # Path to the prediction CSV file
# check_image_and_prediction_size(image_path, prediction_csv_path)

import numpy as np
predictions = np.random.randint(0, 2, (2, 3, 78))
#predictions = np.array([[[1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,1,1,1,
#1,0,1,1,0,1,0,0,0,1,0,1,0,1,1,1,1,0,1,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,1,0,1,1,0,1,0,1,0,1,0,0,0]]])
output_dir = "./to_delete"

# Call the function to generate the submission file
# submission.generate_submission_file(predictions, output_dir)

import PlanktonDataset
# test = PlanktonDataset.PlanktonDataset(dir='/mounts/Datasets3/2024-2025-ChallengePlankton/test/', patch_size=256, train=False)
# print(test[1])

from torch.utils.data import DataLoader

def calculate_class_proportions(dataset, batch_size=32):
    positive_count = 0
    negative_count = 0

    # Use DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for img_patch, mask_patch in dataloader:
        # Flatten mask and calculate positive/negative counts
        mask_flat = mask_patch.view(-1)
        positive_count += (mask_flat == 1).sum().item()
        negative_count += (mask_flat == 0).sum().item()
    
    return positive_count, negative_count

# Example usage:
train_dataset = PlanktonDataset.PlanktonDataset(dir="/mounts/Datasets3/2024-2025-ChallengePlankton/train", patch_size=256, train=True)
positive_count, negative_count = calculate_class_proportions(train_dataset)

print(f"Positive count: {positive_count}")
print(f"Negative count: {negative_count}")






