import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

dimensions = 2
hidden_size_1 = 256
hidden_size_2 = 128
hidden_size_3 = 64
encoding_size = 1

# Generate random training data (x-y coordinates)
X = np.random.rand(100000, 2)

# Define encoder from x-y to latent space with BatchNormalization and Dropout
encoder = Sequential()
encoder.add(Input((dimensions,)))
encoder.add(Dense(hidden_size_1))
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))
encoder.add(Dense(hidden_size_2))
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))
encoder.add(Dense(hidden_size_3))
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))
encoder.add(Dense(encoding_size, activation='linear'))

# Define decoder from latent space to x-y with BatchNormalization and Dropout
decoder = Sequential()
decoder.add(Input((encoding_size,)))
decoder.add(Dense(hidden_size_3))
decoder.add(BatchNormalization())
decoder.add(Activation('relu'))
decoder.add(Dense(hidden_size_2))
decoder.add(BatchNormalization())
decoder.add(Activation('relu'))
decoder.add(Dense(hidden_size_1))
decoder.add(BatchNormalization())
decoder.add(Activation('relu'))
decoder.add(Dense(dimensions, activation='linear'))

# Combine encoder and decoder into the autoencoder
auto_input_layer = Input(shape=(dimensions,))
encoded = encoder(auto_input_layer)
decoded = decoder(encoded)
autoencoder = Model(auto_input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = autoencoder.fit(X, X, epochs=300, batch_size=256)

# Show a summary
autoencoder.summary()
decoder.save('decoder.keras')
encoder.save('encoder.keras')
autoencoder.save('model.keras')
