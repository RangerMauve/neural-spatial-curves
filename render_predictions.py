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

# Set default background color to black
plt.rcParams['figure.facecolor'] = '#111111'
plt.rcParams['axes.facecolor'] = '#111111'

# Set default text/axis color to white
plt.rcParams['text.color'] = '#F2F2F2'
plt.rcParams['axes.labelcolor'] = '#F2F2F2'
plt.rcParams['xtick.color'] = '#F2F2F2'
plt.rcParams['ytick.color'] = '#F2F2F2'

# Plot original vs predicted points
plt.scatter(X_test[:, 0], X_test[:, 1], c='#6E2DE5', label='Original')
plt.scatter(predicted[:, 0], predicted[:, 1], c='#2DE56E', label='Predicted')

# Draw lines from test points to predicted points
for i in range(len(X_test)):
    plt.plot([X_test[i, 0], predicted[i, 0]], [
        X_test[i, 1], predicted[i, 1]], 'w--')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Autoencoder Reconstruction')
plt.legend()
plt.savefig('predictions.svg', format='svg')
