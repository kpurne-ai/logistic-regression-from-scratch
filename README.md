# Logistic Regression From Scratch

A Logistic Regression implementation built from scratch using NumPy and Gradient Descent.

## Features

* Binary Classification
* Sigmoid Activation Function
* Binary Cross Entropy Loss
* Gradient Descent Optimization
* Probability Prediction
* Class Prediction
* Loss History Tracking
* Object-Oriented Design

## Project Structure

logistic-regression-from-scratch/

├── model/
│   └── LogisticRegression.py
│
├── utils/
│   ├── loss.py
│   └── metrics.py
│
├── experiments/
│   └── exp1.py
│
├── requirements.txt
│
└── README.md

## Concepts Implemented

* Logistic Regression
* Sigmoid Function
* Binary Classification
* Binary Cross Entropy Loss
* Gradient Descent
* Probability Prediction
* Model Evaluation

## Usage

```python
from model.LogisticRegression import LogisticRegression

model = LogisticRegression(
    learning_rate=0.1,
    n_iterations=1000
)

model.fit(X, y)

probabilities = model.predict_proba(X)

predictions = model.predict(X)
```

## Learning Goals

This project was built to understand the mathematics and implementation behind Logistic Regression without using machine learning libraries such as scikit-learn.

The implementation focuses on:

* Understanding classification
* Understanding probability outputs
* Implementing gradient descent from scratch
* Building reusable machine learning models
* Preparing for Neural Networks and Deep Learning

## Requirements

* Python
* NumPy
* Matplotlib

