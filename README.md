# EEE 443: Neural Networks - Spring 2024 - 2025

## Project Repository

This repository contains solutions for the projects assigned in EEE 443/543 - Spring 2025.

## Projects Overview

### Project 1: Two-Layer Neural Network and Decision Region Estimation

#### Part 1: Neural Network Implementation
- Design a two-layer neural network using the **signum activation function**.
- Implement the logic function:  
  `f(x1, x2, x3) = x1'x2x3 + x1x2'`
- Represent FALSE as `-1` and TRUE as `1`.
- Draw and describe the final network architecture.

#### Part 2: Decision Region Estimation
- Generate **1000 random points** in the range `[-2,2]²`.
- Classify points using the neural network.
- Plot points in **blue** for output `0` and **red** for output `1`.
- Submit the plot and describe the decision boundary.

---

### Project 2: Perceptron Learning Algorithm

#### Task: Perceptron Training Algorithm
- Implement a **perceptron algorithm** with a step activation function.
- **Steps:**
  - Initialize weights `w0, w1, w2` randomly.
  - Generate `100` random data points within `[-1,1]²`.
  - Classify and plot points in two groups.
  - Train perceptron iteratively until convergence.
  - Analyze performance with different learning rates (`η = 1, 10, 0.1`).
  - Repeat with `1000` data points and compare results.

---

## **Project 3: Multicategory Perceptron Training Algorithm for Digit Classification**  

### **Overview**  
In this project, a **Multicategory Perceptron Training Algorithm (PTA)** will be implemented to classify handwritten digits from the **MNIST dataset**.  

### **Steps**  
1. **Download the MNIST dataset** (training/test images and labels).  
2. **Train a perceptron-based neural network** with:
   - **784 input nodes** (one for each pixel in a 28×28 image).  
   - **10 output nodes**, each representing a digit (0-9).  
   - **Weight updates** based on misclassified samples.  
3. **Stop training** when the misclassification rate falls below a chosen threshold (ϵ).  
4. **Evaluate the trained model** on the MNIST test set.  
5. **Experiment with different dataset sizes (n = 50, 1000, 60000)** and analyze performance.  

---
 
## **Project 4: Neural Network for Curve Fitting**  

### **Overview**  
This project involves implementing a **1×24×1** neural network to perform curve fitting. The network will be trained using the **backpropagation algorithm with online learning** to minimize **Mean Squared Error (MSE)**.  

### **Steps**  
1. **Data Generation**:  
   - Generate **300** random values **x** in **[0,1]**.  
   - Compute target values **d** using a given function with added noise.  

2. **Neural Network Architecture**:  
   - **Input Layer**: 1 neuron  
   - **Hidden Layer**: 24 neurons (tanh activation)  
   - **Output Layer**: 1 neuron (linear activation)  

3. **Training**:  
   - Optimize weights using **backpropagation**.  
   - Adjust **learning rate (η)** dynamically to ensure stability.  

4. **Evaluation**:  
   - Plot **MSE vs. epochs** during training.  
   - Compare the learned curve to the original data.  

# Project 5: Neural Network for Digit Classification

- **Objective:**  
  Build a neural network from scratch that can classify handwritten digits (0–9) using the MNIST dataset. Write your own implementation of a neural network and the backpropagation algorithm (calculate all derivatives manually). Do not use external libraries for these computations.
 - **Dataset:**  
    Use the MNIST dataset, which includes 60,000 training images and 10,000 test images.
 - **Network Design:**  
  - **Input Layer:** 784 neurons (one for each pixel in a 28×28 image).
  - **Output Layer:**
    - Use 10 neurons with a one-hot encoding for digits (e.g., `[1, 0, ..., 0]` for 0, `[0, 1, ..., 0]` for 1, etc.).
- **Hidden Layers:** Design the network architecture (you decide the number and size of hidden layers).
- **Training:**  
  Train the network using backpropagation to minimize the difference between the network's output and the correct labels.
- **Evaluation:**  
  Test your model on the MNIST test set. The network should achieve at least a 95% accuracy.

# Project 6: Geometry Shape Classification

The goal is to classify grayscale images of geometric shapes into one of nine predefined classes using a convolutional neural network (CNN) built with PyTorch.

## Dataset
The dataset includes 90,000 grayscale images (200x200), with 10,000 images per class:
- Circle, Square, Octagon, Heptagon, Nonagon, Star, Hexagon, Pentagon, Triangle

## Files

- `0601_22001758_TAKAK.py` – Loads data, creates training/testing sets, trains the model.
- `0602_22001758_TAKAK.pt` – Trained model (weights only, ≤ 50MB).
- `0603_22001758_TAKAK.py` – Inference script that prints predicted class labels for all `.png` images in the current directory.

## Model Architecture
A CNN with:
- 2 convolutional blocks
- 3 additional max-pooling layers
- Fully connected layers with dropout
- Cross-entropy loss and Adam optimizer

# Project 7: Character-Level Name Generation

## Overview  
Train a character-level LSTM in PyTorch to generate English given-names.  The network sees sequences of 11 one-hot encoded characters (26 letters + `<EON>` token) and learns to predict the next character.  At inference you supply a start letter and the model samples 20 new names via softmax.

---

## Files  

- `names.txt`  
  – 2 000 lower-case names, one per line.

- `0701_22001758_TAKAK.py`  
  – Preprocessing & training  
  - Reads `names.txt`, pads to length 11 with `<EON>`, builds one-hot arrays.  
  - Defines single-layer LSTM (input 27, hidden 128, output 27).  
  - Trains with cross-entropy loss & Adam (lr=1e-3) for 200 epochs.  
  - Saves loss plot (`lossPlot.png`) and weights (`0702_22001758_TAKAK.pth`).

- `0702_22001758_TAKAK.pth`  
  – Trained model weights.

- `0703_22001758_TAKAK.py`  
  – Inference script  
  - Loads weights, prompts for letter (a–z).  
  - Generates 20 names by autoregressive sampling with temperature=1.

---

## Model Architecture  

1. **Input representation**  
   - Sequence length \(T=11\).  
   - Vocabulary size \(V=27\) (a–z + `<EON>`).  
   - Each character → one-hot vector in \(\mathbb R^{27}\).

2. **LSTM layer**  
   - Single layer, hidden size \(H=128\).  
   - Processes the \(T\)-step input sequence, updating hidden & cell states.

3. **Output projection**  
   - Linear layer: \(ℝ^{128}→ℝ^{27}\).  
   - Produces logits for next‐character prediction at each time step.

4. **Softmax & sampling**  
   - Softmax converts logits to probabilities.  
   - During training: compute cross-entropy loss against true next-character one-hot.  
   - During inference: sample from the softmax distribution to pick the next character.  

# Project 8: Exploring Inference in Large Language Models

## Overview

This project explores various aspects of inference in large language models (LLMs) using the minimal implementation provided in the nanoGPT repository. The assignment focuses on understanding prompt generation, temperature and top-k sampling, token embeddings, and the effect of pruning transformer layers on generated outputs.

---

## Steps

1. **Inference Setup**

  - Clone the repository: https://github.com/karpathy/nanoGPT
  - Run a basic inference command:
  python sample.py --init_from=gpt2 --start="one ring to rule them all," --num_samples=3 --max_new_tokens=100
  - Confirm the model generates valid outputs using the prompt and specified parameters.

2. **Model Analysis**

  - Briefly describe the model’s architecture: the number of transformer layers, the use of activation functions, and how text is generated through autoregressive decoding.
  - Explain the roles of:
    - --start: the initial text prompt
    - --num_samples: the number of sequences generated
    - --max_new_tokens: the length of each generated sequence

3. **Temperature Experiments**

  - Set the temperature to 0.1, generate outputs, and observe the deterministic behavior.
  - Set the temperature to 10, generate outputs, and observe the randomness and incoherence.
  - Set top_k = 2 with temperature = 10, and analyze how restricting the vocabulary impacts generation.
  - Discuss the purpose of the temperature parameter and how it influences the distribution of token selection.

4. **Token Embedding Analysis**

  - Locate the token embedding matrix wte in the model.
  - Choose five common English words (e.g., "dog", "city", "book", "love", "science") and five rare or fabricated tokens (e.g., "xqzt", "blorpt", "thraldor", "uvuvwevwe", "zzzzz").
  - Tokenize each and retrieve their embedding vectors.
  - Compute:
    - The L2 norm of each vector
    - Pairwise cosine similarities among the common tokens
    - Pairwise cosine similarities among the rare tokens
  - Compare and interpret the differences in terms of vector norms and similarities.

5. **Transformer Layer Pruning**

  - Modify the model to use only even-numbered transformer blocks (layers 0, 2, 4, ...).
  - Generate output and compare with the full-depth model.
  - Analyze the impact on coherence, fluency, and length.
  - Repeat the experiment by removing the final half of all layers.
  - Comment on the influence of depth on language understanding and generation.
  





