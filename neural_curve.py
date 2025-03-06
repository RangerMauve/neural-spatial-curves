import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

dimensions = 2
hidden_size_1 = 64
hidden_size_2 = 32
encoding_size = 1


def create_and_train_autoencoder():
    # Generate random training data (x-y coordinates)
    X = np.random.rand(100000, 2)

    # Define the autoencoder model
    input_layer = Input(shape=(dimensions,))
    encoded_hidden_1 = Dense(hidden_size_1, activation='relu')(input_layer)
    encoded_hidden_2 = Dense(
        hidden_size_2, activation='relu')(encoded_hidden_1)
    encoded = Dense(encoding_size, activation='linear')(encoded_hidden_2)

    encoder = Model(input_layer, encoded)

    decoder_input = Input(shape=(encoding_size,))
    decoded_hidden_1 = Dense(hidden_size_2, activation='relu')(decoder_input)
    decoded_hidden_2 = Dense(
        hidden_size_1, activation='relu')(decoded_hidden_1)
    decoded = Dense(dimensions, activation='linear')(decoded_hidden_2)
    decoder = Model(decoder_input, decoded)

    auto_input_layer = Input(shape=(dimensions,))
    autoencoder = Model(auto_input_layer, decoder(encoder(auto_input_layer)))

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = autoencoder.fit(X, X, epochs=300, batch_size=32)

    return autoencoder, encoder, decoder


# Train the autoencoder
autoencoder, encoder, decoder = create_and_train_autoencoder()
autoencoder.summary()
decoder.save('decoder.keras')
encoder.save('encoder.keras')
autoencoder.save('model.keras')
