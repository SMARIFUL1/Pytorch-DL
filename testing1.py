from anomalib.deploy import OpenVINOInferencer
import cv2


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def evaluate_image(image_path, model_path, threshold=0.5):
    # Load the model
    inferencer = OpenVINOInferencer(path=model_path, task="detection")  # Use the task type as a string if needed
    image = load_image(image_path)

    # Perform inference
    prediction = inferencer.predict(image=image)

    # Access the prediction attributes
    predicted_score = prediction.pred_score
    predicted_label = "anomaly" if predicted_score > threshold else "normal"

    print(f'Predicted Label: {predicted_label}, Predicted Score: {predicted_score}')


if __name__ == "__main__":
    model_path = r"C:\Users\smari\PycharmProjects\transfer_learning\anomalib_weight\weights\openvino\model.xml"
    image_path = r"D:\seminar\dataset\test\normal\17112023191642.png"
    evaluate_image(image_path, model_path)
