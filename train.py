import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "C:/Users/Nayeem/Downloads/Data Set/archive/bengali_digits"

IMG_SIZE = 28  # all images resized to 28x28

images = []
labels = []

# Loop through each digit folder
for label in range(10):
    folder_path = os.path.join(DATA_DIR, str(label))
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        # Load image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label)

# Convert to numpy arrays
images = np.array(images) / 255.0  # normalize to 0-1
labels = np.array(labels)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Reshape images for Conv2D: (num_samples, height, width, channels)
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

# Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 output classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10,  # start with 10 epochs for speed
                    batch_size=32)

# Save the trained model
model.save("bangla_digit_model.h5")
print("Model trained and saved as bangla_digit_model.h5")




