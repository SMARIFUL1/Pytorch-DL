import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model.transformer_model import BanknoteTransformer
from utils.dataset import BanknoteDataset
from utils.train_eval import train, evaluate
import torch.nn as nn


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BanknoteDataset("data/banknote", transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = BanknoteTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        loss = train(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1} - Loss: {loss:.4f}, Val Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), 'banknote_transformer.pth')


if __name__ == '__main__':
    main()
