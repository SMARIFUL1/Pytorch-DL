#for directory anomaly testing
from anomalib.deploy import OpenVINOInferencer
import cv2
import os

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def evaluate_image(image_path, model_path, threshold=0.5):
    inferencer = OpenVINOInferencer(path=model_path, task="classification")
    image = load_image(image_path)
    prediction = inferencer.predict(image=image)
    predicted_score = prediction.pred_score
    predicted_label = "anomaly" if predicted_score > threshold else "normal"
    return predicted_label, predicted_score

def evaluate_images_in_directory(model_path, images_dir, threshold=0.5):
    inferencer = OpenVINOInferencer(path=model_path, task="classification")
    image_files = [file for file in os.listdir(images_dir) if file.lower().endswith('.png') or file.lower().endswith('.jpg')]
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        predicted_label, predicted_score = evaluate_image(image_path, model_path, threshold)
        print(f'{image_file}: Predicted Label - {predicted_label}, Predicted Score - {predicted_score:.4f}')

if __name__ == "__main__":
    model_path = r"C:\Users\smari\PycharmProjects\transfer_learning\anomalib_weight\weights\openvino\model.xml"
    images_dir = r"D:\seminar\dataset\test\anomaly"  # Directory containing test images
    evaluate_images_in_directory(model_path, images_dir)
