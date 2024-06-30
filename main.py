class Module:

  def __init__(self, input_size, hidden_units, output_size):
    # initialize weights for input layer
    self.W1 = np.random.randn(input_size, hidden_units) / np.sqrt(input_size)
    self.b1 = np.zeros((1, hidden_units))

    # initialize weights for output layer
    self.W2 = np.random.randn(hidden_units, output_size) / np.sqrt(hidden_units)
    self.b2 = np.zeros((1, output_size))

  def summary(self):
    print(f"Input Size: {self.W1.shape[0]}")
    print(f"Hidden Units: {self.W1.shape[1]}")
    print(f"Output Size: {self.W2.shape[1]}")
    print(self.W1)
    print(self.b1)
    print(self.W2)
    print(self.b2)

  def forward(self, X):
  
    self.z1 = np.dot(X, self.W1) + self.b1
    self.a1 = self.sigmoid(self.z1)
    self.z2 = np.dot(self.a1, self.W2) + self.b2
    self.a2 = self.sigmoid(self.z2)
    return self.a2

  def backward(self, X, y, output, learning_rate):
    m = X.shape[0]
    self.dZ2 = output - y
    self.dW2 = (1 / m) * np.dot(self.a1.T, self.dZ2)
    self.db2 = (1 / m) * np.sum(self.dZ2, axis=0, keepdims=True)
    self.dZ1 = np.dot(self.dZ2, self.W2.T) * self.sigmoid_derivative(self.a1)
    self.dW1 = (1 / m) * np.dot(X.T, self.dZ1)
    self.db1 = (1 / m) * np.sum(self.dZ1, axis=0, keepdims=True)

    self.update_params(learning_rate)

  def update_params(self, learning_rate):
    """

    Args:
      learning_rate:
    """
    self.W1 -= learning_rate * self.dW1
    self.b1 -= learning_rate * self.db1
    self.W2 -= learning_rate * self.dW2
    self.b2 -= learning_rate * self.db2

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def sigmoid_derivative(self, x):
    return x * (1 - x)

  def predict(self, X):
    return self.forward(X)
