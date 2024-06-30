model = Module(input_size=3, hidden_units=128, output_size=4)

epochs = 5000
learning_rate = 1
losses = []

for i in range(epochs):
  y_pred = model.forward(X)

  loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
  losses.append(loss)

  model.backward(X, y, y_pred, learning_rate)

  if (i + 1) % 50 == 0:
    print(f"Epoch {i+1}/{epochs}, Loss: {loss:.4f}")

plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y_pred, axis=1), cmap='viridis')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Model Predictions on Test Set')
plt.show()

# Plot the model's loss curve
plt.plot(range(epochs), losses)
plt.title('Model Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Final predictions and accuracy
predictions = model.predict(X)
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
print(f"\nFinal Accuracy: {accuracy:.4f}")

print("\nSample Predictions:")
print(predictions[:5])
print("\nActual:")
print(y[:5])
