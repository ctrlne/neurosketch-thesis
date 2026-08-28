import os
# Force TensorFlow to use the legacy Keras 2 backend that SHAP would understand
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

# TensorFlow to behave like a good boy like version 1 for SHAP's gradient math
tf.compat.v1.disable_v2_behavior()

def generate_shap_heatmaps():
    print("1. Loading trained ResNet50 model...")
    model = tf.keras.models.load_model(r"data\processed\resnet50_spiral_model.h5", compile=False)

    print("2. Loading sample patients (1 Healthy, 1 Parkinson's)...")
    image_dir = r"data\processed\images"
    df = pd.read_csv(r"data\processed\final_labeled_kinematics.csv")
    
    healthy_id = df[df['target'] == 0].iloc[0]['patient_id']
    pd_id = df[df['target'] == 1].iloc[0]['patient_id']
    
    sample_images = []
    for pid in [healthy_id, pd_id]:
        img_path = os.path.join(image_dir, f"{pid}_canny.png")
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        sample_images.append(img)
        
    sample_images = np.array(sample_images) / 255.0

    print("3. Generating SHAP Explainer (This might take a minute)...")
    explainer = shap.GradientExplainer(model, sample_images)
    
    print("4. Calculating SHAP values for the spirals...")
    shap_values = explainer.shap_values(sample_images)
    
    # Amplify the microscopic gradients so the plotting library can see them
    if isinstance(shap_values, list):
        shap_values = [sv * 10000 for sv in shap_values]
    else:
        shap_values = shap_values * 10000

    print("5. Saving SHAP heatmaps...")
    shap.image_plot(shap_values, sample_images, show=False)
    
    output_path = r"data\processed\shap_clinical_explanation.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Success! Visual heatmaps saved to: {output_path}")

if __name__ == "__main__":
    generate_shap_heatmaps()