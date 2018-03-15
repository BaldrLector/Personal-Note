# Practical Recommendations for Gradient-Based Training of Deep
---
> This is personal note  on [Practical Recommendations for Gradient-Based Training of Deep](https://arxiv.org/abs/1206.5533) 
---

<!-- TOC -->

- [Practical Recommendations for Gradient-Based Training of Deep](#practical-recommendations-for-gradient-based-training-of-deep)
- [1 Introduction](#1-introduction)
    - [1.1 Deep Learning and Greedy Layer-Wise Pretraining](#11-deep-learning-and-greedy-layer-wise-pretraining)
    - [1.2 Denoising and Contractive Auto-Encoders](#12-denoising-and-contractive-auto-encoders)
    - [1.3 Online Learning and Optimization of Generalization Error](#13-online-learning-and-optimization-of-generalization-error)
- [2 Gradients](#2-gradients)
    - [2.1 Gradient Descent and Learning Rate](#21-gradient-descent-and-learning-rate)
    - [2.2 Gradient Computation and Auto-matic Differentiation](#22-gradient-computation-and-auto-matic-differentiation)
- [3 Hyper-Parameters](#3-hyper-parameters)
    - [3.1 Neural Network Hyper-Parameters](#31-neural-network-hyper-parameters)
        - [3.1.1 Hyper-Parameters of the Approximate Optimization](#311-hyper-parameters-of-the-approximate-optimization)
    - [3.2 Hyper-Parameters of the Model and Training Criterion](#32-hyper-parameters-of-the-model-and-training-criterion)
    - [3.3 Manual Search and Grid Search](#33-manual-search-and-grid-search)
        - [3.3.1 General guidance for the exploration of hyper-parameters](#331-general-guidance-for-the-exploration-of-hyper-parameters)
        - [3.3.2 Coordinate Descent and Multi-Resolution Search](#332-coordinate-descent-and-multi-resolution-search)
        - [3.3.3 Automated and Semi-automated Grid Search](#333-automated-and-semi-automated-grid-search)
        - [3.3.4 Layer-wise optimization of hyper-parameters](#334-layer-wise-optimization-of-hyper-parameters)
    - [3.4 Random Sampling of Hyper-Parameters](#34-random-sampling-of-hyper-parameters)
- [4 Debugging and Analysis](#4-debugging-and-analysis)
    - [4.1 Gradient Checking and Con-trolled Overfitting](#41-gradient-checking-and-con-trolled-overfitting)
    - [4.2 Visualizations and Statistics](#42-visualizations-and-statistics)
- [5 Other Recommendations](#5-other-recommendations)
    - [5.1 Multi-core machines, BLAS and GPUs](#51-multi-core-machines-blas-and-gpus)
    - [5.2 Sparse High-Dimensional Inputs](#52-sparse-high-dimensional-inputs)
    - [5.3 Symbolic Variables, Embeddings,Multi-Task Learning and Multi-Relational Learning](#53-symbolic-variables-embeddingsmulti-task-learning-and-multi-relational-learning)
- [6 Open Questions](#6-open-questions)
    - [6.1 On the Added Difficulty of Train-ing Deeper Architectures](#61-on-the-added-difficulty-of-train-ing-deeper-architectures)
    - [6.2 Adaptive Learning Rates and Second-Order Methods](#62-adaptive-learning-rates-and-second-order-methods)

<!-- /TOC -->


# 1 Introduction

## 1.1 Deep Learning and Greedy Layer-Wise Pretraining

## 1.2 Denoising and Contractive Auto-Encoders

## 1.3 Online Learning and Optimization of Generalization Error

# 2 Gradients

## 2.1 Gradient Descent and Learning Rate

- common gradient descent 

>$${\theta_{t+1} \gets \theta_t - \epsilon_t \frac{\partial L(z_t,\theta)}{\partial \theta_t} }$$

- when use mini-batch

>$${\theta_{t+1} \gets \theta_t - \epsilon_t \frac{1}{B} \sum_{t\prime=Bt+1}^{B(t+1)} {\frac{\partial L(z_t\prime,\theta)}{\partial \theta}} }$$

- - B is batch-size, when ${B=1}$, is online gradient descent.when ${B=train set size}$, is standard gradient descent.
- - B increases, matrix-multiplications  accelerate computation.
- - B increases, the number of updates per computation done decreases, down convergence, 
- - B depend on hardware ,common use ${2^n}$. 
> Combining these two opposing effects yields atypical U-curve with a sweet spot at an intermediatevalue of B.
- - finite training set
SGD convergence does not depend on the size of the training set only on the number of updates and the richness of the training distribution.
- - infinit training set
batch method (which updates only after seeing all
the examples) is hopeless.

- - stocastic version could speed up convergence







## 2.2 Gradient Computation and Auto-matic Differentiation

# 3 Hyper-Parameters


## 3.1 Neural Network Hyper-Parameters

### 3.1.1 Hyper-Parameters of the Approximate Optimization

## 3.2 Hyper-Parameters of the Model and Training Criterion


## 3.3 Manual Search and Grid Search

### 3.3.1 General guidance for the exploration of hyper-parameters

### 3.3.2 Coordinate Descent and Multi-Resolution Search

### 3.3.3 Automated and Semi-automated Grid Search

### 3.3.4 Layer-wise optimization of hyper-parameters


## 3.4 Random Sampling of Hyper-Parameters

- Algorithm 1 : Greedy layer-wise hyper-parameter optimization.

# 4 Debugging and Analysis

## 4.1 Gradient Checking and Con-trolled Overfitting

## 4.2 Visualizations and Statistics

# 5 Other Recommendations

## 5.1 Multi-core machines, BLAS and GPUs

## 5.2 Sparse High-Dimensional Inputs

## 5.3 Symbolic Variables, Embeddings,Multi-Task Learning and Multi-Relational Learning

# 6 Open Questions

## 6.1 On the Added Difficulty of Train-ing Deeper Architectures

## 6.2 Adaptive Learning Rates and Second-Order Methods

