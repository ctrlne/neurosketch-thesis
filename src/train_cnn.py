import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def train_thesis_visual_model():
    print("1. Loading visual dataset...")
    df = pd.read_csv(r"data\processed\final_labeled_kinematics.csv")
    image_dir = r"data\processed\images"
    
    X_images, y_labels = [], []
    for index, row in df.iterrows():
        img_path = os.path.join(image_dir, f"{row['patient_id']}_canny.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (256, 256))
            X_images.append(img)
            y_labels.append(row['target'])
            
    X = np.array(X_images) / 255.0  
    y = np.array(y_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("2. Building Approved ResNet50 Architecture...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    
    # STRICTLY FREEZE the entire base model to prevent memorization on a small dataset
    base_model.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    # Heavy 70% Dropout forces the AI to look at the whole spiral, not specific pixels
    x = Dropout(0.7)(x) 
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("3. Training Deep Learning CNN...")
    model.fit(X_train, y_train, epochs=50, batch_size=8, 
              validation_data=(X_test, y_test), callbacks=[early_stop])
    
    model.save(r"data\processed\resnet50_spiral_model.h5")
    print("Classifier B saved!")

if __name__ == "__main__":
    train_thesis_visual_model()