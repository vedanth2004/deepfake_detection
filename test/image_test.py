import tensorflow as tf
import numpy as np
import cv2
import os
import sys

# ====== CONFIG ======
MODEL_NAME = "image.h5"         # model file name
IMAGE_NAME = "download.jpeg"    # test image file name
# =====================

# Get current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full paths for model and image
MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_NAME)
IMAGE_PATH = os.path.join(SCRIPT_DIR, IMAGE_NAME)

# ---- Check existence ----
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    sys.exit(1)
else:
    print(f"✅ Found model: {MODEL_PATH}")

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image file not found: {IMAGE_PATH}")
    sys.exit(1)
else:
    print(f"✅ Found image: {IMAGE_PATH}")

# ---- Load model ----
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ---- Read and preprocess image ----
print("Reading image...")
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"❌ Could not read image from path: {IMAGE_PATH}")

# Convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Match model input size
target_size = model.input_shape[1:3]  # (height, width)
if target_size[0] is not None and target_size[1] is not None:
    img = cv2.resize(img, target_size)

# Convert to float32 and preprocess
img = img.astype("float32")
try:
    img = tf.keras.applications.xception.preprocess_input(img)
except Exception:
    img /= 255.0  # fallback for custom-trained models

# Add batch dimension
img = np.expand_dims(img, axis=0)

# ---- Predict ----
print("Running prediction...")
pred = model.predict(img)

# ---- Interpret output ----
if pred.shape[-1] == 1:  # Binary (sigmoid)
    prob = float(pred[0][0])
else:  # Categorical (softmax)
    prob = float(pred[0][1]) if pred.shape[-1] >= 2 else float(pred[0][0])

label = "DEEPFAKE" if prob >= 0.5 else "REAL"

# ---- Output ----
print("\n=======================")
print(f"Prediction: {label}")
print(f"Deepfake Probability: {prob:.4f}")
print("=======================")
