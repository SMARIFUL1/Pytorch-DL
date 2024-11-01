from anomalib import TaskType
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.deploy import ExportType
from anomalib.callbacks import ModelCheckpoint
from anomalib.data import Folder
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

dataset_root = f'D:\\seminar\\dataset\\train'

def train():
    # Create the datamodule
    datamodule = Folder(
        name="train",
        root=dataset_root,
        normal_dir="down_img",
        abnormal_dir="down_img_anomaly",
        mask_dir="abnormal_mask",  # Add this for segmentation
        train_batch_size=1,
        num_workers=4,
        task=TaskType.SEGMENTATION  # Switch to SEGMENTATION task
    )
    datamodule.setup()

    model = EfficientAd()  # Update model to handle segmentation
    engine = Engine(max_epochs=5, task=TaskType.SEGMENTATION,
                    callbacks=[ModelCheckpoint(dirpath='checkpoint/', every_n_epochs=1, save_last=True)])

    # Train the model
    engine.fit(datamodule=datamodule, model=model)
    engine.export(export_type=ExportType.OPENVINO,
                  model=model,
                  export_root='anomalib_weight')

def visualize_anomalies(image_path, model, datamodule):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare the image as input for the model
    input_tensor = datamodule.test_transform(image=image)["image"]
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Predict anomaly map
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    anomaly_map = output["anomaly_map"].cpu().numpy().squeeze()

    # Normalize anomaly map for visualization
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

    # Overlay the anomaly map on the original image
    heatmap = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train()

