import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Configuration
NUM_CLASSES = 12 # Replace with the exact number from your dataset
BATCH_SIZE = 32
EPOCHS = 15

print("🌱 Starting PyTorch Crop Disease Detector Pipeline...")

# 1. Custom CNN Baseline
class CustomCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 2. Transfer Learning Models (MobileNet & ResNet)
def get_mobilenet(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def get_resnet(num_classes):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 3. Setup Device and Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 Using device: {device}")

custom_cnn = CustomCNN(NUM_CLASSES).to(device)
mobilenet = get_mobilenet(NUM_CLASSES).to(device)
resnet = get_resnet(NUM_CLASSES).to(device)

# 4. Define Loss and Optimizers
criterion = nn.CrossEntropyLoss()
# Example optimizer for MobileNet
optimizer = torch.optim.Adam(mobilenet.classifier.parameters(), lr=0.001)

# 5. Data Augmentation (Image Preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("✅ Models and Data Pipeline successfully initialized.")
print("📌 Note: In a full run, you would load the dataset using torchvision.datasets.ImageFolder")
print("and run the training loop over the specified EPOCHS.")