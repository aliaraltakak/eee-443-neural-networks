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



