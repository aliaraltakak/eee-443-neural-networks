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


  





