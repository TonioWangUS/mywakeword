import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load Preprocessed Data
DATA_PATH = './spectrograms'
data = np.load(os.path.join(DATA_PATH, 'data.npy'))
labels = np.load(os.path.join(DATA_PATH, 'labels.npy'))

# Reshape Data for CNN Input
data = data[..., np.newaxis]  # Add channel dimension

# Split into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 100, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save Model
model.save('wakeword_model.h5')
print("Model saved as wakeword_model.h5")
