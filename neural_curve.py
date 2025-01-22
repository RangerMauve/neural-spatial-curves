from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)


def create_and_train_autoencoder():
    # Generate random training data (x-y coordinates)
    X = np.random.rand(1000, 2)

    # Define the autoencoder model
    input_layer = Input(shape=(2,))
    hidden_layer_1 = Dense(64, activation='relu')(input_layer)
    hidden_layer_2 = Dense(1, activation='linear')(
        hidden_layer_1)  # Bottleneck layer with one neuron
    hidden_layer_3 = Dense(64, activation='relu')(hidden_layer_2)
    output_layer = Dense(2)(hidden_layer_3)  # Output layer

    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = autoencoder.fit(X, X, epochs=500, batch_size=32)

    return autoencoder


# Train the autoencoder
autoencoder = create_and_train_autoencoder()
autoencoder.save('model.keras')


def predict(autoencoder, input_data):
    return autoencoder.predict(input_data)


# Example usage:
if __name__ == "__main__":
    # Generate test data
    X_test = np.random.rand(10, 2)  # 5 samples for testing

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
