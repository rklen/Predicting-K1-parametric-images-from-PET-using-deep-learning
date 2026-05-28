from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

# Read in scaled data (change paths to your real ones)
tacs_training = np.load("path/to/training/data/x_train.npy")
k1_training = np.load("path/to/training/data/y_train.npy")
tacs_validation = np.load("path/to/training/data/x_validate.npy")
k1_validation = np.load("path/to/training/data/y_validate.npy")

keras.backend.clear_session()
tf.random.set_seed(2525)

# Build the model structure
model_k1 = keras.Sequential()
model_k1.add(layers.Dense(40, activation='leaky_relu', input_shape=(tacs_training.shape[1],)))
model_k1.add(layers.Dense(30, activation="leaky_relu"))
model_k1.add(layers.Dense(20, activation="leaky_relu"))
model_k1.add(layers.Dropout(0.1, seed=252525))
model_k1.add(layers.Dense(10, activation="leaky_relu"))
model_k1.add(layers.Dense(1, activation="linear"))

# Compile the model
model_k1.compile(
    loss=keras.losses.MeanAbsoluteError(),
    optimizer=keras.optimizers.Adamax(learning_rate=0.0001),
    metrics=[keras.metrics.MeanAbsoluteError()]
    )

# Train the model
history = model_k1.fit(
    tacs_training,
    k1_training,
    batch_size=32,
    validation_data=(tacs_validation, k1_validation),
    epochs=10,
    shuffle=True
    )

# Save the trained model (run the actual test data from Norway on my own computer)
model_k1.save("Outputs/trained_model.keras")
