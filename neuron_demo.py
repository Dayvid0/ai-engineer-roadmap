import numpy as np

def relu(z):
    return np.maximum(0, z)   # keep positives, turn negatives into 0

# One neuron: weighted sum of inputs, plus a bias, then an activation
x = np.array([2.0, -1.0])          # two input features
w = np.array([0.5, 1.5])           # one weight per input
b = 0.25                           # bias
z = np.dot(w, x) + b               # 0.5*2 + 1.5*(-1) + 0.25
print("weighted sum z =", z)
print("after relu     =", relu(z))

# A layer is several neurons at once: a weight matrix instead of a weight vector
W1 = np.array([[ 1.0, -1.0],       # neuron A's weights
               [ 0.5,  2.0],       # neuron B's weights
               [-1.0,  0.5]])      # neuron C's weights
b1 = np.array([0.0, 0.1, -0.2])
hidden = relu(W1 @ x + b1)
print("\nhidden layer (3 neurons) =", hidden)

# Output layer: one neuron that combines the 3 hidden values into a prediction
W2 = np.array([0.7, -0.4, 1.1])
b2 = 0.05
prediction = W2 @ hidden + b2
print("prediction               =", prediction)
