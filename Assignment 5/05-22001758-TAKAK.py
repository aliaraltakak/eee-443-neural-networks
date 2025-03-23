# Import the required libraries.
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import torch

# Define the file paths to training set and labels.
trainingSetPath = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 5/Training Set/train-images-idx3-ubyte"
trainingLabelsPath = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 5/Training Set/train-labels-idx1-ubyte"

# Define the file paths to test set and labels.
testSetPath = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 5/Test Set/t10k-images-idx3-ubyte"
testLabelPath = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 5/Test Set/t10k-labels-idx1-ubyte"

# Load the training set.
trainImages = idx2numpy.convert_from_file(trainingSetPath)
trainLabels = idx2numpy.convert_from_file(trainingLabelsPath)

# Load the test set.
testImages = idx2numpy.convert_from_file(testSetPath)
testLabels = idx2numpy.convert_from_file(testLabelPath)

# Convert NumPy arrays to PyTorch tensors.
trainImagesTensor = torch.tensor(trainImages, dtype=torch.float32) / 255.0 
trainLabelsTensor = torch.tensor(trainLabels, dtype=torch.long)
testImagesTensor = torch.tensor(testImages, dtype=torch.float32) / 255.0  
testLabelsTensor = torch.tensor(testLabels, dtype=torch.long)

# Print shapes and dimensions.
print("Training images shapes and dimensions: ", trainImages.shape)
print("Training labels shapes and dimensions: ", trainLabels.shape)
print("Testing images shapes and dimensions: ", testImages.shape)
print("Testing labels shapes and dimensions: ", testLabels.shape)

# Flatten the training and test images.
trainImagesTensor = trainImagesTensor.view(trainImagesTensor.size(0), -1)
testImagesTensor = testImagesTensor.view(testImagesTensor.size(0), -1)

# Define a function for one-hot encoding.
def oneHotEncoding(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

# Define activation functions.
def tanhActivation(x):
    return np.tanh(x)

def tanhDerivative(x):
    return 1 - np.tanh(x)**2

def softmaxActivation(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def sigmoidActivation(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    s = sigmoidActivation(x)
    return s * (1 - s)

# Initialize parameters.
def initParameters(inputSize, hiddenSize, outputSize):
    W1 = np.random.randn(hiddenSize, inputSize) * 0.01
    W2 = np.random.randn(outputSize, hiddenSize) * 0.01
    B1 = np.zeros((hiddenSize, 1))
    B2 = np.zeros((outputSize, 1))
    return W1, W2, B1, B2

# Define cross-entropy loss.
def crossEntropyLoss(A2, Y):
    m = Y.shape[1]
    A2 = np.clip(A2, 1e-10, 1.0)  
    loss = -np.sum(Y * np.log(A2)) / m
    return loss

# Forward propagation with dropout (applied only to hidden layer activations).
def forwardPropagation(x, W1, W2, B1, B2, dropout_keep_prob=1.0):
    # Hidden layer.
    Z1 = np.dot(W1, x) + B1
    A1 = tanhActivation(Z1)
    if dropout_keep_prob < 1.0:
        
        dropout_mask = (np.random.rand(*A1.shape) < dropout_keep_prob) / dropout_keep_prob
        A1 *= dropout_mask
    
    Z2 = np.dot(W2, A1) + B2
    A2 = softmaxActivation(Z2)
    return Z1, A1, Z2, A2

# Backpropagation remains the same.
def backPropagation(x, Y, Z1, A1, Z2, A2, W2):
    m = Y.shape[1]
    dZ2 = A2 - Y  
    dW2 = np.dot(dZ2, A1.T) / m
    dB2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * tanhDerivative(Z1)
    dW1 = np.dot(dZ1, x.T) / m
    dB1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, dB1, dW2, dB2

# Parameter update with momentum.
def updateParametersWithMomentum(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learningRate, 
                                 vW1, vB1, vW2, vB2, momentum):
    # Update velocity terms.
    vW1 = momentum * vW1 + learningRate * dW1
    vB1 = momentum * vB1 + learningRate * dB1
    vW2 = momentum * vW2 + learningRate * dW2
    vB2 = momentum * vB2 + learningRate * dB2
    # Update parameters.
    W1 = W1 - vW1
    B1 = B1 - vB1
    W2 = W2 - vW2
    B2 = B2 - vB2
    return W1, B1, W2, B2, vW1, vB1, vW2, vB2

# Prepare the data.
X_train = trainImagesTensor.numpy().T
Y_train = oneHotEncoding(trainLabelsTensor, num_classes=10).numpy().T 
X_test = testImagesTensor.numpy().T 
Y_test = oneHotEncoding(testLabelsTensor, num_classes=10).numpy().T  

# Hyperparameters.
maxEpochs = 250
initialLearningRate = 0.25
learningRate = initialLearningRate
dropout_keep_prob = 0.85  
momentum_coeff = 0.9

# Initialize parameters.
inputSize = 784 
hiddenSize = 175
outputSize = 10 
W1, W2, B1, B2 = initParameters(inputSize, hiddenSize, outputSize)

# Initialize momentum velocities as zeros.
vW1 = np.zeros_like(W1)
vB1 = np.zeros_like(B1)
vW2 = np.zeros_like(W2)
vB2 = np.zeros_like(B2)

# Lists to store metrics.
trainLosses = []
testLosses = []
trainErrors = []
testErrors = []
trainAccuracies = []
testAccuracies = []

numTrain = X_train.shape[1]

for epoch in range(maxEpochs):
    
    # Forward propagation on training set with dropout.
    Z1, A1, Z2, A2 = forwardPropagation(X_train, W1, W2, B1, B2, dropout_keep_prob)
    train_loss = crossEntropyLoss(A2, Y_train)
    train_predictions = np.argmax(A2, axis=0)
    train_true = np.argmax(Y_train, axis=0)
    train_error = numTrain - np.sum(train_predictions == train_true)
    train_accuracy = np.mean(train_predictions == train_true)
    
    # Backpropagation.
    dW1, dB1, dW2, dB2 = backPropagation(X_train, Y_train, Z1, A1, Z2, A2, W2)
    
    # Update parameters using momentum.
    W1, B1, W2, B2, vW1, vB1, vW2, vB2 = updateParametersWithMomentum(
        W1, B1, W2, B2, dW1, dB1, dW2, dB2, learningRate, vW1, vB1, vW2, vB2, momentum_coeff)
    
    # Re-evaluate on training set (without dropout).
    Z1_train, A1_train, Z2_train, A2_train = forwardPropagation(X_train, W1, W2, B1, B2, dropout_keep_prob=1.0)
    train_loss = crossEntropyLoss(A2_train, Y_train)
    train_predictions = np.argmax(A2_train, axis=0)
    train_true = np.argmax(Y_train, axis=0)
    train_error = numTrain - np.sum(train_predictions == train_true)
    train_accuracy = np.mean(train_predictions == train_true)
    
    # Evaluate on test set.
    Z1_test, A1_test, Z2_test, A2_test = forwardPropagation(X_test, W1, W2, B1, B2, dropout_keep_prob=1.0)
    test_loss = crossEntropyLoss(A2_test, Y_test)
    test_predictions = np.argmax(A2_test, axis=0)
    test_true = np.argmax(Y_test, axis=0)
    test_error = X_test.shape[1] - np.sum(test_predictions == test_true)
    test_accuracy = np.mean(test_predictions == test_true)
    
    # Append metrics.
    trainLosses.append(train_loss)
    testLosses.append(test_loss)
    trainErrors.append(train_error)
    testErrors.append(test_error)
    trainAccuracies.append(train_accuracy)
    testAccuracies.append(test_accuracy)
    
    # Print metrics.
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Errors = {train_error}, Train Accuracy = {train_accuracy:.4f}, " \
          f"Test Loss = {test_loss:.4f}, Test Errors = {test_error}, Test Accuracy = {test_accuracy:.4f}")
    
    # Dynamic Learning Rate Adaptation.
    if epoch > 0 and trainLosses[-1] > trainLosses[-2]:
        learningRate *= 0.9
        print(f"    Learning rate decreased to {learningRate:.4f} due to increased training loss.")
    
    # Early stopping if training error rate < 1%.
    if train_error / numTrain < 0.01:
        print(f"Early stopping at epoch {epoch} as training error rate is below 1%.")
        break

# Plotting metrics.
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(trainErrors, label="Train Errors")
plt.plot(testErrors, label="Test Errors")
plt.title("Epoch vs. Classification Errors")
plt.xlabel("Epoch")
plt.ylabel("Number of Errors")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(trainLosses, label="Train Loss")
plt.plot(testLosses, label="Test Loss")
plt.title("Epoch vs. Energy (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Energy (Loss)")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(trainAccuracies, label="Train Accuracy")
plt.plot(testAccuracies, label="Test Accuracy")
plt.title("Epoch vs. Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
