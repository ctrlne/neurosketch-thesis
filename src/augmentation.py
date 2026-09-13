import cv2
import os
import pandas as pd
import numpy as np

def augment_visuals(image_dir):
    print("Augmenting Visual Stream...")
    for img_name in os.listdir(image_dir):
        if not img_name.endswith("_pressure.png"): continue
        
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Geometric Augmentation: Mild Rotations (+15 and -15 degrees)
        rows, cols = img.shape
        matrix_cw = cv2.getRotationMatrix2D((cols/2, rows/2), -15, 1)
        matrix_ccw = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
        
        img_cw = cv2.warpAffine(img, matrix_cw, (cols, rows))
        img_ccw = cv2.warpAffine(img, matrix_ccw, (cols, rows))
        
        base_name = img_name.replace(".png", "")
        cv2.imwrite(os.path.join(image_dir, f"{base_name}_rot_cw.png"), img_cw)
        cv2.imwrite(os.path.join(image_dir, f"{base_name}_rot_ccw.png"), img_ccw)
        
def augment_kinematics(csv_path):
    print("Augmenting Kinematic Stream...")
    df = pd.read_csv(csv_path)
    
    # synthetic noise injection (Mild Gaussian Noise)
    augmented_rows = []
    for index, row in df.iterrows():
        noise_factor = 0.05 # 5% clinical variation
        
        new_row_1 = row.copy()
        new_row_2 = row.copy()
        
        for col in ['mean_velocity', 'pressure_variance', 'jerk_variance', 'tremor_power_4_to_6_hz']:
            std_dev = df[col].std()
            new_row_1[col] += np.random.normal(0, std_dev * noise_factor)
            new_row_2[col] -= np.random.normal(0, std_dev * noise_factor)
            
        new_row_1['patient_id'] = str(row['patient_id']) + "_aug1"
        new_row_2['patient_id'] = str(row['patient_id']) + "_aug2"
        
        augmented_rows.extend([new_row_1, new_row_2])
        
    aug_df = pd.DataFrame(augmented_rows)
    final_df = pd.concat([df, aug_df], ignore_index=True)
    
    augmented_csv_path = csv_path.replace(".csv", "_augmented.csv")
    final_df.to_csv(augmented_csv_path, index=False)
    print(f"Expanded kinematic dataset to {len(final_df)} rows!")

if __name__ == "__main__":
    PROCESSED_IMAGES = r"data\processed\images"
    MASTER_CSV = r"data\processed\kinematic_features_master.csv"
    
    augment_visuals(PROCESSED_IMAGES)
    augment_kinematics(MASTER_CSV)
    print("Data Expansion Complete!")