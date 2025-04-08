# Import the required libraries.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Random seed.
studentID = 22001758
np.random.seed(studentID)

# Set device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Define the shape classes
shapeClasses = ['Circle', 'Square', 'Octagon', 'Heptagon', 'Nonagon', 'Star', 'Hexagon', 'Pentagon', 'Triangle']

# Mapping from class index to label
idx_to_label = {idx: label for idx, label in enumerate(sorted(shapeClasses))}

# Define the ConvBlock.
class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)

# Define the CNN model.
class CNNClassifier(nn.Module):
    def __init__(self, numClasses):
        super(CNNClassifier, self).__init__()
        self.convBlock1 = ConvBlock(1, 32)    
        self.convBlock2 = ConvBlock(32, 64)    
        self.pool1 = nn.MaxPool2d(2)           
        self.pool2 = nn.MaxPool2d(2)           
        self.pool3 = nn.MaxPool2d(2)           
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, numClasses)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the transform.
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

# Load the trained model.
MODEL_PATH = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 6/0602-22001758-TAKAK.pt"
model = CNNClassifier(numClasses=9).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Set the test image folder.
testFolder = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 6/output"

# Loop through the files.
for fileName in sorted(os.listdir(testFolder)):
    if fileName.lower().endswith(".png"):
        imagePath = os.path.join(testFolder, fileName)
        image = Image.open(imagePath).convert("L")
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            prediction = output.argmax(dim=1).item()
            predicted_label = idx_to_label[prediction]

        print(f"{fileName}: {predicted_label}")

