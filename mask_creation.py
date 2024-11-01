import cv2
import numpy as np
import os

# Paths
image_dir = 'D:\\seminar\\dataset\\train\\down_img_anomaly'  # Path to abnormal images
mask_dir = 'D:\\seminar\\dataset\\train\\abnormal_mask'  # Path to save generated masks

# Create the mask directory if it doesn't exist
os.makedirs(mask_dir, exist_ok=True)

# Function to create masks
def create_mask(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to highlight anomalies
    # Threshold value might need adjustment depending on your images
    _, binary_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)

    # Use morphological operations to clean up the mask (optional)
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return clean_mask

# Iterate over each abnormal image
for image_name in os.listdir(image_dir):
    # Load the image
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)

    # Create a mask for the anomalies
    mask = create_mask(image)

    # Save the mask
    mask_path = os.path.join(mask_dir, image_name)
    cv2.imwrite(mask_path, mask)
