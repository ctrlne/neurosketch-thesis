import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def execute_fusion():
    print("1. Loading Multi-Modal Datasets...")
    df = pd.read_csv(r"data\processed\final_labeled_kinematics.csv")
    image_dir = r"data\processed\images"
    
    kin_features = ['mean_velocity', 'pressure_variance', 'jerk_variance', 'tremor_power_4_to_6_hz']
    X_kin = df[kin_features]
    y = df['target']
    
    X_vis = []
    for patient_id in df['patient_id']:
        patient_str = str(patient_id)
        
        # Route directly to the pressure-mapped images
        if patient_str.endswith("_aug1"):
            base_id = patient_str.replace("_aug1", "")
            img_path = os.path.join(image_dir, f"{base_id}_pressure_rot_cw.png")
        elif patient_str.endswith("_aug2"):
            base_id = patient_str.replace("_aug2", "")
            img_path = os.path.join(image_dir, f"{base_id}_pressure_rot_ccw.png")
        else:
            img_path = os.path.join(image_dir, f"{patient_str}_pressure.png")
            
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        X_vis.append(img)
        
    # Standardize image array to match ResNet50 expectations
    X_vis = preprocess_input(np.array(X_vis))
    
    X_kin_train, X_kin_test, X_vis_train, X_vis_test, y_train, y_test = train_test_split(
        X_kin, X_vis, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("2. Generating Kinematic Probabilities (Classifier A)...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_kin_train, y_train)
    rf_prob_train = rf_model.predict_proba(X_kin_train)[:, 1]
    rf_prob_test = rf_model.predict_proba(X_kin_test)[:, 1]
    print(f"   -> Random Forest Standalone: {accuracy_score(y_test, rf_model.predict(X_kin_test)) * 100:.2f}%")
    
    print("3. Generating Visual Probabilities (Classifier B)...")
    # Load the correct .h5 model saved from your 97% training run
    cnn_model = tf.keras.models.load_model(r"data\processed\resnet50_spiral_model.h5")
    cnn_prob_train = cnn_model.predict(X_vis_train, verbose=0).flatten()
    cnn_prob_test = cnn_model.predict(X_vis_test, verbose=0).flatten()
    cnn_preds = (cnn_prob_test >= 0.5).astype(int)
    print(f"   -> ResNet50 Standalone: {accuracy_score(y_test, cnn_preds) * 100:.2f}%")
    
    print("4. Executing Late Fusion Strategy (Meta-Learner)...")
    X_meta_train = np.column_stack((rf_prob_train, cnn_prob_train))
    X_meta_test = np.column_stack((rf_prob_test, cnn_prob_test))
    
    meta_learner = LogisticRegression()
    meta_learner.fit(X_meta_train, y_train)
    final_predictions = meta_learner.predict(X_meta_test)
    
    print("\n=== FINAL NEUROSKETCH FUSION RESULTS ===")
    print(f"Fused Accuracy: {accuracy_score(y_test, final_predictions) * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, final_predictions, target_names=["Healthy (0)", "Parkinson's (1)"]))

if __name__ == "__main__":
    execute_fusion()