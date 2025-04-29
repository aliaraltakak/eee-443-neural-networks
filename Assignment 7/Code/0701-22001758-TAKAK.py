# Import the required libraries.
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define a function to load and preprocess the name dataset.
def loadNames(filePath):
    names = []
    with open(filePath) as f:
        for line in f:
            entry = line.strip().lower()
            if entry:
                names.append(entry)
    return names

# Define a function to build the vocabulary mappings.
def buildVocab():
    vocabList     = ["<EON>"] + [chr(c) for c in range(ord("a"), ord("z")+1)]
    charToIndex   = {ch: i for i, ch in enumerate(vocabList)}
    indexToChar   = {i: ch for ch, i in charToIndex.items()}
    return vocabList, charToIndex, indexToChar

# Define a function to convert names into input/output arrays.
def preprocessNames(nameList, charToIndex, maxLen):
    vocabSize  = len(charToIndex)
    count      = len(nameList)
    inputArray = np.zeros((count, maxLen, vocabSize), dtype=np.float32)
    targetArray= np.zeros((count, maxLen),       dtype=np.int64)
    for i, name in enumerate(nameList):
        paddedName = list(name) + ["<EON>"]*(maxLen - len(name))
        for t, ch in enumerate(paddedName):
            inputArray[i, t, charToIndex[ch]] = 1.0
            nextCh = paddedName[t+1] if t+1 < maxLen else "<EON>"
            targetArray[i, t] = charToIndex[nextCh]
    return inputArray, targetArray

# Define a function to wrap arrays into a PyTorch DataLoader.
def getDataLoader(inputArray, targetArray, batchSize):
    tensorX = torch.tensor(inputArray)
    tensorY = torch.tensor(targetArray)
    dataSet = TensorDataset(tensorX, tensorY)
    return DataLoader(dataSet, batch_size=batchSize, shuffle=True)

# Define the LSTM model class for name generation.
class NameLSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers, numClasses):
        super().__init__()
        self.lstm = nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True)
        self.fc   = nn.Linear(hiddenSize, numClasses)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

# Define a function to train the model and return loss history.
def trainModel(model, dataLoader, numEpochs, learningRate, device):
    model.to(device)
    optimizer    = torch.optim.Adam(model.parameters(), lr=learningRate)
    lossFn       = nn.CrossEntropyLoss()
    totalBatches = len(dataLoader)
    totalSamples = len(dataLoader.dataset)
    lossHistory  = []

    for epoch in range(1, numEpochs+1):
        model.train()
        runningLoss = 0.0

        for batchIdx, (batchX, batchY) in enumerate(dataLoader, start=1):
            batchX, batchY = batchX.to(device), batchY.to(device)
            logits = model(batchX)
            loss   = lossFn(logits.view(-1, logits.size(-1)), batchY.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runningLoss += loss.item() * batchX.size(0)

            # Compute and display percentage of this epoch completed.
            percent = batchIdx / totalBatches * 100
            print(f"\rEpoch {epoch:02d}/{numEpochs} [{percent:5.1f}%]", end="")

        averageLoss = runningLoss / totalSamples
        lossHistory.append(averageLoss)
        # Overwrite the progress line with final loss for the epoch.
        print(f"\rEpoch {epoch:02d}/{numEpochs} completed — loss: {averageLoss:.4f}")

    return lossHistory

# Define a function to save model weights to disk.
def saveModel(model, filePath):
    torch.save(model.state_dict(), filePath)

# Define a function to plot and save the training loss curve.
def plotLoss(lossHistory, filePath):
    plt.figure()
    plt.plot(lossHistory)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss")
    plt.savefig(filePath)
    plt.close()

# Define file paths and training hyperparameters.
dataFile     = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 7/names.txt"
modelFile    = "0702-22001758-TAKAK.pth"
plotFile     = "lossPlot.png"
batchSize    = 16
numEpochs    = 200
learningRate = 1e-3
maxLen       = 11  
studentID = 22001758
np.random.seed(studentID)

# Main execution block.
if __name__ == "__main__":
    names         = loadNames(dataFile)
    vocabList, charToIndex, indexToChar = buildVocab()
    inputs, targets= preprocessNames(names, charToIndex, maxLen)
    loader        = getDataLoader(inputs, targets, batchSize)
    model         = NameLSTM(inputSize=len(vocabList),
                             hiddenSize=128,
                             numLayers=1,
                             numClasses=len(vocabList))
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history       = trainModel(model, loader, numEpochs, learningRate, device)
    saveModel(model, modelFile)
    plotLoss(history, plotFile)
    print("Training complete.")
