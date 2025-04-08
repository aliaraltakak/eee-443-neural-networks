# Import the required libraries.
import os
import pickle
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Random seed.
studentID = 22001758
np.random.seed(studentID)

# Define the path to image folder.
imageFolder = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 6/output"

# Define a function to create train and test sets.
def createDatasetPickleFiles(imageFolder, outputTrainFile="trainingSet.file", outputTestFile="testingSet.file"):

    # Define the shape classes.
    shapeClasses = ['Circle', 'Square', 'Octagon', 'Heptagon', 'Nonagon', 'Star', 'Hexagon', 'Pentagon', 'Triangle']
    
    # Create a dictionary to hold image paths per class.
    classImages = defaultdict(list)
    
    # Group files by class name found in the filename.
    for fileName in os.listdir(imageFolder):
        for shape in shapeClasses:
            if shape.lower() in fileName.lower():
                classImages[shape].append(os.path.join(imageFolder, fileName))
                break
    
    trainData = []
    testData = []
    
    # Sort each class's image list and split into train/test.
    for shape in shapeClasses:
        files = sorted(classImages[shape])
        trainFiles = files[:8000]
        testFiles = files[8000:10000]
        
        trainData.extend([(filePath, shape) for filePath in trainFiles])
        testData.extend([(filePath, shape) for filePath in testFiles])
    
    # Save using pickle.
    with open(outputTrainFile, "wb") as trainFile:
        pickle.dump(trainData, trainFile)
    
    with open(outputTestFile, "wb") as testFile:
        pickle.dump(testData, testFile)
    
    print(f"Saved '{outputTrainFile}' and '{outputTestFile}'")

# Uncomment to separate.
#createDatasetPickleFiles(imageFolder = imageFolder)

# Define the neural network architecture.
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

# Define the dataset class.
class GeometryDataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(label for _, label in self.data)))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label_idx = self.label_to_idx[label]
        return image, label_idx
    
# Configure training environment and variables.
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 9
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "0602-22001758-TAKAK.pt"
print(f"Using available device: {DEVICE}")

# Define the transforms.
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

#  Define the paths and looad the data.
trainingPath = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 6/trainingSet.file"
testingPath = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 6/testingSet.file"

trainDataset = GeometryDataset(trainingPath, transform=transform)
testDataset = GeometryDataset(testingPath, transform=transform)

trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the training process.
model = CNNClassifier(NUM_CLASSES).to(DEVICE)  
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss()

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

# Training phase.
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    loop = tqdm(trainLoader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for data, target in loop:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        loop.set_postfix(loss=loss.item())

    train_loss = total_loss / len(trainLoader)
    train_acc = correct / len(trainLoader.dataset)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Model evaluation.
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

    test_loss /= len(testLoader)
    test_acc = correct / len(testLoader.dataset)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1} finished: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, "
          f"Train Acc={train_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%")

    scheduler.step()

# Save the trained model parameters.
model_half = model.half()
torch.save(model_half.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved as {MODEL_SAVE_PATH}")

# Plotting.
# Loss versus Epoch plot.
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch vs Loss Graph")
plt.legend()
plt.savefig("Epoch vs Loss.png")

# Accuracy versus Epoch plot.
plt.figure()
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Epoch vs Accuracy Graph")
plt.legend()
plt.savefig("Epoch vs Accuracy.png")