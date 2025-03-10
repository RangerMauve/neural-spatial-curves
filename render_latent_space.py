from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

model = load_model('decoder.keras')


def predict(model, input_data):
    return model.predict(input_data)


# Create an array of 100 numbers from 0 to 1.0 in 0.01 increments
input_data = np.arange(-1.00, 1.00, 0.1)

print(input_data)

# Call predict on the array to get back a list of x-y coordinates
coords = predict(model, input_data)

# Set default background color to black
plt.rcParams['figure.facecolor'] = '#111111'
plt.rcParams['axes.facecolor'] = '#111111'

# Set default text/axis color to white
plt.rcParams['text.color'] = '#F2F2F2'
plt.rcParams['axes.labelcolor'] = '#F2F2F2'
plt.rcParams['xtick.color'] = '#F2F2F2'
plt.rcParams['ytick.color'] = '#F2F2F2'

# Use pyplot to render the coordinates in a graph called "Latent Space"
plt.scatter(coords[:, 0], coords[:, 1], color='#6E2DE5')

# Connect the points with a line
for i in range(len(coords) - 1):
    plt.plot([coords[i, 0], coords[i + 1, 0]],
             [coords[i, 1], coords[i + 1, 1]], 'w--')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Decoder Latent Space')

# Save the plot to `latent_space.svg`
plt.savefig('latent_space.svg')
