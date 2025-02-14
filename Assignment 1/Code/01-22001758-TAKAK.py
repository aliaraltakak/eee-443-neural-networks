# Import the required libraries.
import numpy as np
import matplotlib.pyplot as plt

# Define the step activation function.
def stepActivation(x):
    return np.heaviside(x,1)

# Define a perceptron class.
class perceptronANN:

    def __init__(self, Weights: np.ndarray, activationFunction = "step"):
        
        # Define the activation function abbreviations. 
        actFunctions = {
            "step": stepActivation,
        }

        self.activationFunction = actFunctions[activationFunction]
        self.Weights = Weights
        self.neuronInputs = len(Weights) - 1

    # Define a method to evaluate the output of the perceptron given its inputs.
    def fit(self, inputs: np.array):

        outputValue = np.dot(np.insert(inputs, 0, 1), self.Weights)

        return self.activationFunction(outputValue)

# Define a class for the feedforward neural network.
class FeedforwardNN:

    def __init__(self, neuronsPerLayer: list, Weights: list, activationFunction = "step"):

        self.inputCount = len(Weights[0]) - 1
        self.outputCount = neuronsPerLayer[-1]
        self.layerCount = len(neuronsPerLayer)
        self.activationFunction = activationFunction

        # Construct the neural network. 
        self.Neurons =  []
        weightsIndex = 0

        # Loop through the layers.
        for i in range(len(neuronsPerLayer)):
            self.Neurons.append([])

            # Loop through the neurons in the layer.
            for j in range(neuronsPerLayer[i]):
                self.Neurons[i].append( perceptronANN(Weights = Weights[weightsIndex], activationFunction = activationFunction))
                weightsIndex = weightsIndex + 1


    # Define a method to predict the outcome of the network.
    def fit(self, inputs: np.ndarray):

        # Iterate through the layers for forward propagation.
        for i in range(len(self.Neurons)):
            
            # Use the initial inputs for the first layer.
            if i == 0:
                arrayInput = np.array(inputs)
            
            # Use the outputs of the previous layer.
            else:
                arrayInput = np.array(lastOutputs)
            
            lastOutputs = []

            for j in range(len(self.Neurons[i])):
                lastOutputs.append(self.Neurons[i][j].fit(arrayInput))
        
        return np.array(lastOutputs)

if __name__ == "__main__":

    # Define the given weights of the network.
    networkWeights = [
        [1, 1, -1], # Hidden Neuron 1 with bias = 1, x = 1, y = -1.
        [1, -1, -1], # Hidden Neuron 2 with bias = 1, x = -1, y = -1.
        [0, -1, 0], # Hidden Neuron 3 with bias = 0, x = -1, y = 0.
        [-1.5, 1, 1, -1] # Output neuron bias = -1.5 and Weights for H1, H2, H3.
    ]

    # Define the feedforward neural network.
    ProjectNN = FeedforwardNN(neuronsPerLayer = [3, 1], Weights = networkWeights)

    # Generate 1000 random points in the space [-2, 2]^2.
    randPoints = np.random.uniform(low=-2, high=2, size=(1000, 2))
    plotOutputs = []

    for element in range(1000):
        plotOutputs.append(ProjectNN.fit(randPoints[element, :])[0])
    
    arrayOutputs = np.array(plotOutputs)
    
    # Create the plot for the classified points on the space.
    redPoints = randPoints[arrayOutputs == 1, :]
    bluePoints = randPoints[arrayOutputs == 0, :]
    
    plt.figure()
    plt.plot(redPoints[:, 0], redPoints[:, 1], 'ro', label = "Classified output = 1")
    plt.plot(bluePoints[:, 0], bluePoints[:, 1], 'bo', label = "Classified output = 0")
    plt.legend()
    plt.xlabel("X Input")
    plt.ylabel("Y Input")
    plt.title("Classification output of the Neural Network")
    plt.show()
