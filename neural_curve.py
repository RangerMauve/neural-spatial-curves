import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Set the number of CPU cores to use for training
# I do `CPU count - 2` to let other processes run
cpu_count = 14  # You can adjust this based on your system's capabilities
tf.config.set_visible_devices(
    tf.config.list_physical_devices('CPU')[:cpu_count], 'CPU')

dimensions = 2
hidden_size_1 = 128
hidden_size_2 = 64
encoding_size = 1

# Generate random training data (x-y coordinates)
X = np.random.rand(100000, 2)

# Define encoder from x-y to latent space with BatchNormalization
encoder = Sequential()
encoder.add(Input((dimensions,)))
encoder.add(Dense(hidden_size_1))
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))
encoder.add(Dense(hidden_size_2))
encoder.add(BatchNormalization())
encoder.add(Activation('relu'))
encoder.add(Dense(encoding_size, activation='linear'))

# Define decoder from latent space to x-y with BatchNormalization
decoder = Sequential()
decoder.add(Input((encoding_size,)))
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
