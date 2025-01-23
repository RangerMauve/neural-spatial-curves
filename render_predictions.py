from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

autoencoder = load_model('model.keras')

print("Loaded model")


def predict(autoencoder, input_data):
    return autoencoder.predict(input_data)


# Generate test data
X_test = np.random.rand(300, 2)  # 5 samples for testing

# Use the trained autoencoder to make predictions
predicted = predict(autoencoder, X_test)

# Plot original vs predicted points
plt.scatter(X_test[:, 0], X_test[:, 1], c='blue', label='Original')
plt.scatter(predicted[:, 0], predicted[:, 1], c='red', label='Predicted')

# Draw lines from test points to predicted points
for i in range(len(X_test)):
    plt.plot([X_test[i, 0], predicted[i, 0]], [
        X_test[i, 1], predicted[i, 1]], 'g--')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Autoencoder Reconstruction')
plt.legend()
plt.savefig('predictions.svg', format='svg')
