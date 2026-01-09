import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("bangla_digit_model.h5")

# Directory containing test images
TEST_DIR = "C:/Users/Nayeem/python_learning/bangla_digit_nn/test_images"

# Get a few sample images (say 9)
sample_images = os.listdir(TEST_DIR)[:9]  # first 9 images
images = []
filenames = []

for img_name in sample_images:
    img_path = os.path.join(TEST_DIR, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img_resized = cv2.resize(img, (28,28)) / 255.0
        images.append(img_resized.reshape(28,28,1))
        filenames.append(img_name)

images = np.array(images)

# Predict
preds = model.predict(images)
pred_labels = np.argmax(preds, axis=1)
confidences = np.max(preds, axis=1)

# Plot images with predictions
plt.figure(figsize=(10,10))

for i in range(len(images)):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i].reshape(28,28), cmap='gray')
    plt.title(f"{pred_labels[i]} ({confidences[i]:.2f})")
    plt.axis('off')

plt.suptitle("Figure 3: Sample Test Images with Predicted Labels and Confidence Scores", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
