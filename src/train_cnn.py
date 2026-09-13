import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def train_clinical_visual_model():
    print("1. Loading visual dataset...")
    df = pd.read_csv(r"data\processed\final_labeled_kinematics.csv")
    image_dir = r"data\processed\images"
    
    X_images, y_labels = [], []
    for index, row in df.iterrows():
        patient_str = str(row['patient_id'])
        
        # Route the augmented CSV labels to the correct physical image files
        if patient_str.endswith("_aug1"):
            base_id = patient_str.replace("_aug1", "")
            img_path = os.path.join(image_dir, f"{base_id}_pressure_rot_cw.png")
        elif patient_str.endswith("_aug2"):
            base_id = patient_str.replace("_aug2", "")
            img_path = os.path.join(image_dir, f"{base_id}_pressure_rot_ccw.png")
        else:
            img_path = os.path.join(image_dir, f"{patient_str}_pressure.png")
            
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (256, 256))
            X_images.append(img)
            y_labels.append(row['target'])
            
    # CRITICAL FIX: Use ResNet50's native preprocessing instead of / 255.0
    X = preprocess_input(np.array(X_images))
    y = np.array(y_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("2. Initializing Dynamic Image Generator...")
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    print("3. Building Thesis-Aligned ResNet50...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x) 
    x = Dropout(0.5)(x) 
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    print("4. Training Deep Learning CNN...")
    model.fit(datagen.flow(X_train, y_train, batch_size=8), 
              epochs=50, validation_data=(X_test, y_test), 
              callbacks=[early_stop, reduce_lr])
    
    model.save(r"data\processed\resnet50_spiral_model.h5")
    print("Clinical Classifier B saved!")

if __name__ == "__main__":
    train_clinical_visual_model()