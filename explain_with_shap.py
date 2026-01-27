import os
import glob
import numpy as np
import shap
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = 'yolov8n.pt'
IMAGE_DIR = 'dataset_kart/images/test'
OUTPUT_DIR = 'shap_explanations'
NUM_IMAGES = 3
TARGET_SIZE = (320, 320)
CLASS_NAMES = ['go_kart']
TARGET_CLASS_INDEX = 0

# --- 1. Setup Environment ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"SHAP explanation images will be saved in: {OUTPUT_DIR}")

# --- 2. Load Model ---
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# --- 3. Load and Preprocess Images ---
image_paths = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))[:NUM_IMAGES]
if not image_paths:
    print(f"Error: No JPG images found in {IMAGE_DIR}. Please check the path.")
    exit()

print(f"Loading {len(image_paths)} images from {IMAGE_DIR}...")
original_images = [cv2.imread(p) for p in image_paths]
resized_images = [cv2.resize(img, TARGET_SIZE) for img in original_images]
stacked_images = np.stack(resized_images)

print("Images loaded and resized successfully.")

# --- 4. Define the SHAP Prediction Function ---
def f(x):
    """
    SHAP prediction function.
    x: numpy array of images with shape (N, H, W, 3) and RGB channel order.
    """
    all_scores = []
    
    for i in range(x.shape[0]):
        img_rgb = x[i]
        img_bgr = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

        results = model(img_bgr, verbose=False)
        
        highest_confidence = 0.0
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            target_detections = [b for b in boxes if int(b.cls) == TARGET_CLASS_INDEX]
            if target_detections:
                highest_confidence = max([float(b.conf) for b in target_detections])

        all_scores.append([highest_confidence])
        
    return np.array(all_scores)

print("SHAP prediction function defined.")

# --- 5. Run SHAP Explainer ---
print("Initializing SHAP explainer...")
masker = shap.maskers.Image("inpaint_telea", stacked_images[0].shape)
explainer = shap.Explainer(f, masker, output_names=CLASS_NAMES)

print("Calculating SHAP values... This can take several minutes.")
shap_values = explainer(
    stacked_images,
    max_evals=1000,
    batch_size=10
)

print("SHAP values calculated.")

# --- 6. Plot and Save Explanations ---
print(f"Saving {len(shap_values)} plots...")
for i in tqdm(range(len(shap_values))):
    # The .values attribute of an Explanation object holds the numpy array.
    # We extract the values for the 'go_kart' class (index 0).
    shap_values_for_plot = shap_values[i].values[:,:,0]

    shap.image_plot(
        shap_values=[shap_values_for_plot],
        pixel_values=stacked_images[i],
        show=False
    )
    
    output_path = os.path.join(OUTPUT_DIR, f"shap_{os.path.basename(image_paths[i])}")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

print(f"Successfully saved {len(image_paths)} explanation images in '{OUTPUT_DIR}'.")
print("Script finished.")
