# Multilayer Perceptron

Multilayer Perceptron is a neural network coded from scratch in Python.

## Table of content
___

* [Introduction](#introduction)
* [Tehnologies](#technologies)
* [Launch](#launch)
* [Options](#options)
* [Examples](#examples)

## Introduction
___

This project is a school 42 project coded with [projectKB](https://github.com/ProjectKB).

The project is a lite Neural Network library with data processing, modular initialization and optimization options.
Everything is coded with `numpy` with matrixes calculus and accelerated by `numba`.

With `seed 101`, loss is under `0.07` and accuracy is near `99%` with the given dataset in **data**.

## Technologies
___

The project is entirely coded in Python3.8 with some libraries:
- Pandas
- Numpy
- Numba
- Matplotlib
- Scipy

## Options
___

### Arguments

See all arguments in usage by specifying `-h` while executing the code

- provide a seed (numpy)
- provide a model
- provide a file name for the model when saved
- provide dataset (train - test)
- add verbose
- plot metrics

### Data Processing

- Normalisation
- Standardisation
- Split and shuffle dataset (Train - Validation - Test)
- KNN imputer (replace _None_ or _null_ value with the average of the _k-nearest neighboor_ value selected) **Same result as [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)**

### Modular Initialization

- input dimension
- layers size (size of the list defines the number of layers, numbers provided defines the neuron number in layers)
- number of epochs
- learning rate

### Metrics

- Cross Entropy Loss (Train - Validation)
- Mean Squared Error
- Accuracy
- Confusion Matrix
- F1 score

### Activation

- Sigmoid
- Tanh
- Relu
- Leaky Relu
- Parametric Relu

### Regularisation

- L1 (lasso regression)
- L2 (ridge regression)

### Optimization

- Vanilla
- Momentum
- RMSprop
- Adam

## Examples
___

See in **srcs** (`perceptron_train.py` and `perceptron_predict.py`)
