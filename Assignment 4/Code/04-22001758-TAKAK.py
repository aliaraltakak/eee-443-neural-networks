# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np

# Embed random seed as student ID for exact same results cross-platform.
np.random.seed(22001758)

# Define the tanh activation function.
def tanhActivation(x):
    return np.tanh(x)

# Define the derivative of the tanh activation function.
def tanhActivationDerivative(x):
    return 1 - np.tanh(x) ** 2

# Draw n = 300 real numbers uniformly at random on [0, 1].
x = np.random.uniform(0, 1, 300)

# Draw n = 300 real numbers uniformly at random on [-1/10, 1/10].
v = np.random.uniform(-0.1, 0.1, 300)

# Define the provided sinusoidal function.
d = np.sin(20 * x) + 3 * x + v

# Generate a scatterplot for the distribution of the points.
plt.figure(figsize=(8, 6))
plt.scatter(x, d, color='blue', alpha=0.6, label="Data Points (xi, di)")
plt.xlabel("x")
plt.ylabel("d")
plt.title("Scatter Plot of Generated Data Points")
plt.legend()
plt.grid(True)
plt.show()

# Define the neural network parameters.
N = 24
learningRate = 0.01
epochs = 5000

# Initialize the weights and biases.
W1 = np.random.randn(N, 1) * 0.1 # Defines the weights from input to hidden layer.
W2 = np.random.randn(1, N) * 0.1 # Defines the weights from hidden layer to output layer.

B1 = np.zeros((N, 1)) # Bias terms of the hidden layer.
B2 = np.zeros((1, 1)) # Bias term of the output layer.

# Define the training loop with backpropagation algorithm.
meanSquareHist = []

for epoch in range(epochs):

    mse = 0

    for i in range(300):
        # Define the forward pass.
        xi = x[i].reshape(1, 1)
        di = d[i].reshape(1, 1)

        # Hidden layer linear combinations
        Z1 = np.dot(W1, xi) + B1
        A1 = tanhActivation(Z1)

        # Output layer linear combinations.
        Z2 = np.dot(W2, A1) + B2
        yi = Z2

        # Compute the forward pass error.
        forwardPassError = di - yi
        mse = mse + forwardPassError ** 2

        # Define the backpropagation for error minimization.
        dW2 = -2 * forwardPassError * A1.T
        dB2 = -2 * forwardPassError

        dZ1 = np.dot(W2.T, -2 * forwardPassError) * tanhActivationDerivative(Z1)
        dW1 = np.dot(dZ1, xi.T)
        dB1 = dZ1

        # Update the weights and biases accordingly.
        W1 = W1 - (learningRate * dW1)
        W2 = W2 - (learningRate * dW2)

        B1 = B1 - (learningRate * dB1)
        B2 = B2 - (learningRate * dB2)
    
    # Compute the average mean-squared error.
    mse = mse / 300
    meanSquareHist.append(mse[0,0])

    # Add adaptive learning rate adjustment if MSE increases.
    if epoch > 0 and meanSquareHist[-1] > meanSquareHist[-2]:

        learningRate = 0.9 * learningRate


# Generate a plot for mean-squared error over epochs.
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), meanSquareHist, color='red', label="MSE")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training Progress: MSE vs Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Plot the fitted curve.
sortedX = np.linspace(0, 1, 300).reshape(-1, 1)  
Z1 = np.dot(W1, sortedX.T) + B1 
A1 = tanhActivation(Z1)  
f_x = np.dot(W2, A1) + B2 

plt.figure(figsize=(8, 6))
plt.scatter(x, d, color='blue', alpha=0.6, label="Data Points (xi, di)")
plt.plot(sortedX, f_x.T, color='red', linewidth=2, label="Fitted Curve f(x, w0)")
plt.xlabel("x")
plt.ylabel("d")
plt.title("Curve Fitting using Neural Network")
plt.legend()
plt.grid(True)
plt.show()