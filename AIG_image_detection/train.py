import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_utils import ResNetClassifier, train_model
import torch

root = os.getcwd()
TRAIN_DIR = os.path.join(root, "train")
print(TRAIN_DIR)
TEST_DIR = os.path.join(root, "test")
print(TEST_DIR)
BATCH_SIZE = 16
NUM_CLASSES = 2
EPOCHS = 10
LEARNING_RATE = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: " + device)

def prepare_data(train_dir, test_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    train_loader, test_loader, classes = prepare_data(TRAIN_DIR, TEST_DIR, BATCH_SIZE)
    print(f"Classes: {classes}")

    model = ResNetClassifier(num_classes=NUM_CLASSES).to(device)
    print("Call model successfully")

    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        save_path="best_model.pth"
    )
