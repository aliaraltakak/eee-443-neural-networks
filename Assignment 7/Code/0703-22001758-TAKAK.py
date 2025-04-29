# Import the required libraries.
import torch
import numpy as np
import torch.nn.functional as F

# Define file path for the saved model.
modelFile    = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 7/Code/0702-22001758-TAKAK.pth"
maxGenerate  = 20  
numNames     = 20  

# Define a function to rebuild vocabulary mappings.
def buildVocab():
    vocabList   = ["<EON>"] + [chr(c) for c in range(ord("a"), ord("z")+1)]
    charToIndex = {ch: i for i, ch in enumerate(vocabList)}
    indexToChar = {i: ch for ch, i in charToIndex.items()}
    return vocabList, charToIndex, indexToChar

# Define the LSTM model class matching the training architecture.
class NameLSTM(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers, numClasses):
        super().__init__()
        # LSTM layer for sequence modeling.
        self.lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True)
        # Fully-connected layer to project LSTM outputs to character logits.
        self.fc   = torch.nn.Linear(hiddenSize, numClasses)
    def forward(self, x):
        # Pass input through LSTM.
        out, _ = self.lstm(x)
        # Project to vocabulary size.
        return self.fc(out)

# Define a function to load model weights and prepare for inference.
def loadModel(filePath, vocabSize, device):
    model = NameLSTM(inputSize=vocabSize, hiddenSize=128, numLayers=1, numClasses=vocabSize).to(device)
    model.load_state_dict(torch.load(filePath, map_location=device))
    model.eval()
    return model

# Define a function to create a one-hot vector.
def oneHot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec

# Define a function to generate a single name from a start character.
def generateName(model, startChar, charToIndex, indexToChar,
                 maxLen, temperature, device):
    vocabSize = len(charToIndex)
    # initialize sequence of indices
    seq  = [charToIndex[startChar]] + [charToIndex["<EON>"]] * (maxLen-1)
    name = startChar

    with torch.no_grad():
        for t in range(maxLen-1):
            # build one-hot input entirely in PyTorch
            seqTensor = torch.tensor(seq, device=device)                   
            x         = F.one_hot(seqTensor, num_classes=vocabSize).float()  
            x         = x.unsqueeze(0)                                     

            logits = model(x)[0, t]
            probs  = torch.softmax(logits/temperature, dim=0).cpu().numpy()
            nxt    = np.random.choice(vocabSize, p=probs)

            if indexToChar[nxt] == "<EON>":
                break
            name += indexToChar[nxt]
            seq[t+1] = nxt

    return name

# Define a function to generate multiple names for a given start character.
def generateMany(model, charToIndex, indexToChar, startChar, count, **kwargs):
    print(f"\nGenerating {count} names starting with '{startChar}':")
    for _ in range(count):
        print(generateName(model, startChar, charToIndex, indexToChar, **kwargs))

# Main execution block.
if __name__ == "__main__":

    # Rebuild vocabulary.
    vocabList, charToIndex, indexToChar = buildVocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model for inference.
    model  = loadModel(modelFile, vocabSize=len(vocabList), device=device)

    # Prompt user for a starting letter.
    startLetter = input("Enter a starting letter (a–z): ").strip().lower()
    if len(startLetter) != 1 or startLetter not in charToIndex or startLetter == "<EON>":
        print("Invalid input. Please enter a single letter a–z.")
    else:
        # Generate and display 20 names beginning with the provided letter.
        generateMany(
            model,
            charToIndex,
            indexToChar,
            startChar=startLetter,
            count=numNames,
            maxLen=maxGenerate,
            temperature=1.0,
            device=device
        )
