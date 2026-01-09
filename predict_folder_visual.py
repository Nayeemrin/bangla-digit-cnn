import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("bangla_digit_model.h5")

# Folder containing test images
TEST_FOLDER = "C:/Users/Nayeem/python_learning/bangla_digit_nn/test_images"

# Desired display window size
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

# Function to predict a single image
def predict_digit(image_path):
    # Load in grayscale for prediction
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image not found or cannot read: {image_path}")
        return None, None, None

    # Resize for model input
    img_resized = cv2.resize(img, (28,28))
    img_normalized = img_resized / 255.0
    img_input = img_normalized.reshape(1,28,28,1)
    
    # Predict
    pred = model.predict(img_input)
    digit = np.argmax(pred)
    confidence = np.max(pred)
    
    return digit, confidence, img

# Loop through all images in the folder
for file_name in os.listdir(TEST_FOLDER):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        file_path = os.path.join(TEST_FOLDER, file_name)
        digit, confidence, img = predict_digit(file_path)
        if digit is not None: 
            print(f"{file_name} -> Predicted Digit: {digit}, Confidence: {confidence:.2f}")
            
            # Convert grayscale to BGR for display
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Resize image to fixed window size for visualization
            img_large = cv2.resize(img_bgr, (400, 400), interpolation=cv2.INTER_NEAREST)
            
            # Put predicted digit text on image
            cv2.putText(img_large, f"{digit} ({confidence:.2f})", (10,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            
            # Show image
            cv2.imshow("Prediction", img_large)
            key = cv2.waitKey(0)  # Wait for key press
            if key == 27:  # Press ESC to exit early
                break

cv2.destroyAllWindows()
