import numpy as np
import matplotlib.pyplot as plt

def generate_complex_dataset(n_samples=1000, noise_level=0.1):
    np.random.seed(42)
    
    # Generate input features
    t = np.linspace(0, 4*np.pi, n_samples)
    x1 = np.sin(t) + noise_level * np.random.randn(n_samples)
    x2 = np.cos(t) + noise_level * np.random.randn(n_samples)
    x3 = np.sin(2*t) + noise_level * np.random.randn(n_samples)
    
    X = np.column_stack((x1, x2, x3))
    
    # Generate target values
    y1 = (x1 > 0) & (x2 > 0)
    y2 = (x2 > 0) & (x3 > 0)
    y3 = (x3 > 0) & (x1 > 0)
    y4 = ~(y1 | y2 | y3)
    
    y = np.column_stack((y1, y2, y3, y4)).astype(int)
    
    return X, y

# Generate the dataset
X, y = generate_complex_dataset()

# Visualize the dataset
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], c=y.argmax(axis=1), cmap='viridis')
plt.title('X1 vs X2')
plt.xlabel('X1')
plt.ylabel('X2')

plt.subplot(132)
plt.scatter(X[:, 1], X[:, 2], c=y.argmax(axis=1), cmap='viridis')
plt.title('X2 vs X3')
plt.xlabel('X2')
plt.ylabel('X3')

plt.subplot(133)
plt.scatter(X[:, 2], X[:, 0], c=y.argmax(axis=1), cmap='viridis')
plt.title('X3 vs X1')
plt.xlabel('X3')
plt.ylabel('X1')

plt.tight_layout()
plt.show()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nFirst few samples of X:")
print(X[:5])
print("\nFirst few samples of y:")
print(y[:5])
