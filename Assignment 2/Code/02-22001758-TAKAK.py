# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt

# Set my student ID as the random seed so multiple runs will provide the same result.
np.random.seed(22001758)

# Initialize weights randomly.
w0 = np.random.uniform(-0.25, 0.25)
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)

# Print the random weights for visualization.
print("Random weights are: ", w0, w1, w2)

# Generate n = 100 random points in [-1, 1]^2.
points = np.random.uniform(-1, 1, (100, 2))

# Define the class labels.
S1 = []  # Positive class.
S0 = []  # Negative class.

# Determine the points and their classes.
for x1, x2 in points:
    decisionValue = w0 + w1 * x1 + w2 * x2

    if decisionValue >= 0:
        S1.append((x1, x2))
    else:
        S0.append((x1, x2))

# Convert S1 and S0 into NumPy arrays.
S1 = np.array(S1)
S0 = np.array(S0)

# Generate a plot for the decision boundaries and classifications.
xValues = np.linspace(-1, 1, 100)
yValues = (-w0 - w1 * xValues) / w2  # Decision boundary equation.

plt.figure(figsize=(7,7))
plt.plot(xValues, yValues, 'k-', label="Decision Boundary")

if len(S1) > 0:
    plt.scatter(S1[:, 0], S1[:, 1], marker='o', color='b', label="S1: Positive Class")
if len(S0) > 0:
    plt.scatter(S0[:, 0], S0[:, 1], marker='o', color='r', label="S0: Negative Class")

plt.xlabel("x1 as x-axis")
plt.ylabel("x2 as y-axis")
plt.title("Initial Perceptron Classification Results")
plt.legend()
plt.grid()
plt.show()


# Define the perceptron training algorithm.
def perceptronTrainingAlgorithm(points, S1, S0, learningRate=1, maxEpochs=1000):

    # Initialize weights randomly in the range [-1,1].
    w0Prime = 0.400
    w1Prime = -0.623
    w2Prime = -0.12783

    misclassificationCount = []

    # Training loop for the algorithm.
    for epoch in range(maxEpochs):
        misclassified = 0

        for x1, x2 in points:
            weightedSum = w0Prime + w1Prime * x1 + w2Prime * x2
            yPredicted = 1 if weightedSum >= 0 else 0

            trueSum = w0 + w1 * x1 + w2 * x2
            yTrue = 1 if trueSum >= 0 else 0

            if yPredicted != yTrue:
                w0Prime = w0Prime + learningRate * (yTrue - yPredicted) * 1
                w1Prime = w1Prime + learningRate * (yTrue - yPredicted) * x1
                w2Prime = w2Prime + learningRate * (yTrue - yPredicted) * x2
                misclassified += 1

        misclassificationCount.append(misclassified)

        if misclassified == 0:
            break

    return [w0Prime, w1Prime, w2Prime], misclassificationCount

# Define varying learning rates.
learningRates = [1, 10, 0.1]
results = {}

# Test with varying learning rates.
for eta in learningRates:

    trainedWeights, misclassificationResults = perceptronTrainingAlgorithm(points, S1, S0, learningRate=eta)
    results[eta] = (trainedWeights, misclassificationResults)

# Plot misclassification count per epoch for each learning rate.
plt.figure(figsize=(8, 6))

for eta, (_, misclassificationResults) in results.items():
    plt.plot(range(len(misclassificationResults)), misclassificationResults, marker='o', linestyle='-', label=f"η = {eta}")

plt.xlabel("Epoch Number")
plt.ylabel("Number of Misclassifications")
plt.title("Misclassification Trend for Different Learning Rates")
plt.legend()
plt.grid()
plt.show()


# Experiment with n = 1000.
nLarge = 1000
pointsLarge = np.random.uniform(-1, 1, (nLarge, 2))
S1_large = []
S0_large = []

# Classify the points based on the initial decision boundary.
for x1, x2 in pointsLarge:
    decisionValue = w0 + w1 * x1 + w2 * x2
    if decisionValue >= 0:
        S1_large.append((x1, x2))
    else:
        S0_large.append((x1, x2))

S1_large = np.array(S1_large)
S0_large = np.array(S0_large)

# Train perceptron for each learning rate.
results_large = {}

for eta in learningRates:
    trainedWeights_large, misclassificationResults_large = perceptronTrainingAlgorithm(
        pointsLarge, S1_large, S0_large, learningRate=eta
    )
    results_large[eta] = (trainedWeights_large, misclassificationResults_large)

# Plot misclassification count trend for each learning rate (n = 1000).
plt.figure(figsize=(8, 6))

colors = {1: "g", 10: "r", 0.1: "b"} 

for eta, (_, misclassificationResults_large) in results_large.items():
    plt.plot(range(len(misclassificationResults_large)), misclassificationResults_large,
             marker='o', linestyle='-', color=colors[eta], label=f"η = {eta}")

plt.xlabel("Epoch Number")
plt.ylabel("Number of Misclassifications")
plt.title("Misclassification Trend for n = 1000 with Different Learning Rates")
plt.legend()
plt.grid()
plt.show()

# Print final weights for each learning rate experiment.
for eta, (trainedWeights_large, _) in results_large.items():
    print(f"Final weights for n=1000, η={eta}: {trainedWeights_large}")
    print("")
