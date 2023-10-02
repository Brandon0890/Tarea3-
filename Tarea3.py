import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Funciones de activación y sus derivadas
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# 2. Inicialización de pesos de la red
def initialize_weights(layers):
    weights = {}
    for i in range(1, len(layers)):
        weights[i] = np.random.randn(layers[i], layers[i-1]) * 0.01
    return weights

# 3. Feedforward
def feedforward(X, weights):
    activations = {0: X}
    zs = {}
    for i in range(1, len(weights) + 1):
        z = np.dot(activations[i-1], weights[i].T)
        zs[i] = z
        activation = sigmoid(z)
        activations[i] = activation
    return activations, zs

# 4. Retropropagación
def backpropagate(activations, zs, weights, y):
    m = y.shape[0]
    gradients = {}
    L = len(activations) - 1
    dz = activations[L] - y
    dw = np.dot(dz.T, activations[L-1]) / m
    gradients[L] = dw
    for i in range(L-1, 0, -1):
        dz = np.dot(dz, weights[i+1]) * sigmoid_prime(zs[i])
        dw = np.dot(dz.T, activations[i-1]) / m
        gradients[i] = dw
    return gradients

# Entrenamiento
def train(X, y, layers, epochs, lr):
    weights = initialize_weights(layers)
    for epoch in range(epochs):
        activations, zs = feedforward(X, weights)
        gradients = backpropagate(activations, zs, weights, y)
        for i in range(1, len(weights) + 1):
            weights[i] -= lr * gradients[i]
    return weights

# Leemos los datos
data = pd.read_csv("concentlite.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Entrenar
layers = [X.shape[1], 10, 1]  # por ejemplo, una capa oculta de 10 neuronas
weights = train(X, y, layers, 10000, 0.01)

# Predicción
def predict(X, weights):
    activations, _ = feedforward(X, weights)
    return activations[len(layers) - 1]

y_pred = predict(X, weights)
plt.scatter(X[:, 0], X[:, 1], c=y_pred.reshape(-1), cmap='viridis')
plt.show()
