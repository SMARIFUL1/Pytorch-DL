import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from model.transformer_model import BanknoteTransformer

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image(path):
    image = Image.open(path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(image_path, model_path='banknote_transformer.pth', threshold=0.4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BanknoteTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = load_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
        prob = F.softmax(output, dim=1)[0, 1].item()
        print(f"Anomaly Probability: {prob:.4f}")
        print("Anomaly Detected" if prob > threshold else "Normal Image")

if __name__ == "__main__":
    predict("data/fortest/104.png")
