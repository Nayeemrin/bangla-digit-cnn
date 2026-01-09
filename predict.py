import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("bangla_digit_model.h5")

# Function to predict a single image
def predict_digit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28))
    img = img / 255.0
    img = img.reshape(1,28,28,1)
    pred = model.predict(img)
    digit = np.argmax(pred)
    confidence = np.max(pred)
    print(f"Predicted Digit: {digit}, Confidence: {confidence:.2f}")


predict_digit("C:/Users/Nayeem/python_learning/bangla_digit_nn/test_images/test 12.jpg")