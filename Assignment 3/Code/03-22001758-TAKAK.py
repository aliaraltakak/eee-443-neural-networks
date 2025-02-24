# Import the required libraries.
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import torch

# Define the file paths to training set and labels (adjust accordingly).
trainingSetPath = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 3/Training Set/train-images.idx3-ubyte"
trainingLabelsPath = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 3/Training Set/train-labels.idx1-ubyte"

# Define the file paths to testing set and labels (adjust accordingly).
testingSetPath = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 3/Testing Set/t10k-images.idx3-ubyte"
testingLabelsPath = "/Users/aral/Documents/Bilkent Archive/EEE 443 - Neural Networks/Assignment 3/Testing Set/t10k-labels.idx1-ubyte"

# Load the training and testing set.
trainImages = idx2numpy.convert_from_file(trainingSetPath)
trainLabels = idx2numpy.convert_from_file(trainingLabelsPath)
testImages = idx2numpy.convert_from_file(testingSetPath)
testLabels = idx2numpy.convert_from_file(testingLabelsPath)

# Convert NumPy arrays to PyTorch tensors.
trainImagesTensor = torch.tensor(trainImages, dtype=torch.float32) / 255.0 
trainLabelsTensor = torch.tensor(trainLabels, dtype=torch.long)
testImagesTensor = torch.tensor(testImages, dtype=torch.float32) / 255.0  
testLabelsTensor = torch.tensor(testLabels, dtype=torch.long)

# Print shapes and dimensions before hand to ensure that the dataset is properly loaded.
print("Training images shapes and dimensions: ", trainImages.shape)
print("Training labels shapes and dimensions: ", trainLabels.shape)
print("Testing images shapes and dimensions: ", testImages.shape)
print("Testing labels shapes and dimensions: ", testLabels.shape)

# Define a function to visualize the dataset.
def displayImages(images, labels, numberOfDisplays = 10):
    
    fig, axes = plt.subplots(1, numberOfDisplays, figsize = (15, 3))

    for i in range(numberOfDisplays):
        axes[i].imshow(images[i], cmap = "gray")
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis("off")
    
    plt.show()

# Define a function for one-hot encoding.
def oneHotEncoding(labels, num_classes = 10):
    return torch.eye(num_classes)[labels]

# Define a function for training multicategory perceptron.
def trainMultiCategoryPerceptron(trainingImg, trainingLabel, n = 50, eta = 1.0, epsilon = 0.0, maxEpochs = 150):
    
    # Flatten the images.
    flattenedImg = trainingImg.view(trainingImg.shape[0], -1)

    # Execute one-hot encoding for labels.
    oneHotLabels = oneHotEncoding(trainingLabel, num_classes = 10)

    # Initialize random weights.
    W = torch.randn(10, 784)

    # Define the training loop.
    epoch = 0
    errors = []

    while True:

        errorCount = 0

        for i in range(n):

            x_i = flattenedImg[i]
            yTrue = oneHotLabels[i]

            # Compute the local induced fields.
            v = W @ x_i

            # Make the prediction.
            yPred = torch.argmax(v)

            # Check if misclassification occured.
            if yPred != trainingLabel[i]:  

                errorCount = errorCount + 1

                # Update the weights accordingly.
                yTrueVector = yTrue.view(-1, 1)  # Reshape to (10,1)
                x_i_vec = x_i.view(1, -1)  # Reshape to (1,784)
                W += eta * (yTrueVector - (v == torch.max(v)).float().view(-1,1)) @ x_i_vec

        errors.append(errorCount)

        # Print progress
        print(f"Epoch {epoch}: Errors = {errorCount}")

        # Check stopping condition
        if errorCount / n <= epsilon:
            break

                
        if epoch >= maxEpochs:
            print(f"Reached max epochs ({maxEpochs}), stopping training.")
            break

        epoch += 1

    return W, errors

# Define a function to test the multicategory perceptron.
def testMulticategoryPerceptron(W, testImg, testLabel):
    
    # Flatten the test images.
    flattenedTest = testImg.view(testImg.shape[0], -1)

    # Initialize the test errors.
    testErrors = 0

        # Testing Loop
    for i in range(testImg.shape[0]): 
        xTest = flattenedTest[i]  
        vTest = W @ xTest  

        yPredTest = torch.argmax(vTest)  
        
        if yPredTest != testLabel[i]:  
            testErrors = testErrors + 1

    # Compute and return test misclassification rate
    testErrorRate = testErrors / testImg.shape[0]
    print(f"Test Misclassification Rate: {testErrorRate * 100:.2f}%")

    return testErrorRate

# Display examples from both sets.
displayImages(trainImages, trainLabels) # Training set.
displayImages(testImages, testLabels) # Testing set.

# Define different training cases.
trainingCases = [
    {"n": 50, "eta": 1.0, "epsilon": 0.0, "description": "n=50, η=1, ϵ=0"},
    {"n": 1000, "eta": 1.0, "epsilon": 0.0, "description": "n=1000, η=1, ϵ=0"},
    {"n": 60000, "eta": 1.0, "epsilon": 0.0, "description": "n=60000, η=1, ϵ=0"},
    {"n": 60000, "eta": 0.5, "epsilon": 0.01, "description": "n=60000, η=0.5, ϵ=0.01"} 
]

# Store the results.
caseResults = {}

# Loop through each training case
for case in trainingCases:
    print("\n" + "="*50)
    print(f"Training for {case['description']}")
    print("="*50)

    # Train the model
    W_trained, error_history = trainMultiCategoryPerceptron(
        trainImagesTensor, trainLabelsTensor, n=case["n"], eta=case["eta"], epsilon=case["epsilon"]
    )

    # Test the trained model
    test_error = testMulticategoryPerceptron(W_trained, testImagesTensor, testLabelsTensor)

    # Store results
    caseResults[case["description"]] = {"W": W_trained, "error_history": error_history, "test_error": test_error}

    # Plot training misclassification errors over epochs
    plt.figure()
    plt.plot(range(len(error_history)), error_history, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Misclassification Errors")
    plt.title(f"Training Convergence ({case['description']})")
    plt.show()