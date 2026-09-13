import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from preprocessing import NeuroSketchPreprocessor

def run_batch():
    RAW_DIR = r"data\raw\data\PaHaW\PaHaW_public"
    PROCESSED_DIR = r"data\processed"
    os.makedirs(f"{PROCESSED_DIR}/images", exist_ok=True)

    processor = NeuroSketchPreprocessor()
    master_kinematic_data = []

    print("Starting automated batch processing for all patients...")

    # Loop through every patient folder automatically
    for patient_folder in os.listdir(RAW_DIR):
        patient_path = os.path.join(RAW_DIR, patient_folder)
        if not os.path.isdir(patient_path): continue
        
        for file in os.listdir(patient_path):
            # STRICT FILTER: Only grab the first trial of the spiral task
            if file.endswith("__1_1.svc"):
                svc_path = os.path.join(patient_path, file)
                patient_id = file.replace(".svc", "")
                
                # 1. Load Data
                df = pd.read_csv(svc_path, sep=r'\s+', skiprows=1, 
                                 names=['y', 'x', 'time', 'button', 'azimuth', 'altitude', 'pressure'])
                
                # 2. Extract Kinematics & Save to Master List
                kinematics = processor.process_kinematics(df)
                kinematics['patient_id'] = patient_id
                master_kinematic_data.append(kinematics)
                
                # 3. Generate Pressure-Sensitive Image
                df_pen = df[df['button'] == 1]
                
                # Normalize pressure values to dynamically scale the ink thickness
                p_min, p_max = df_pen['pressure'].min(), df_pen['pressure'].max()
                scaled_pressure = (df_pen['pressure'] - p_min) / (p_max - p_min + 1e-5)
                
                plt.figure(figsize=(5,5), facecolor='white')
                # Draw the spiral using pressure for the stroke size (s)
                plt.scatter(df_pen['x'], -df_pen['y'], c='black', s=scaled_pressure * 30, cmap='gray')
                plt.axis('off')
                
                # Save directly to the images folder, bypassing Canny Edge entirely
                final_img_path = f"{PROCESSED_DIR}/images/{patient_id}_pressure.png"
                plt.savefig(final_img_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                
                edges = processor.process_visuals(final_img_path)
                cv2.imwrite(f"{PROCESSED_DIR}/images/{patient_id}_canny.png", edges)

    # Export a single clean CSV for the Random Forest model
    final_df = pd.DataFrame(master_kinematic_data)
    final_df.to_csv(f"{PROCESSED_DIR}/kinematic_features_master.csv", index=False)
    
    # Cleanup temp file
    if os.path.exists(f"{PROCESSED_DIR}/temp.png"):
        os.remove(f"{PROCESSED_DIR}/temp.png")
        
    print(f"Batch complete! Processed {len(final_df)} files.")
    print("Images saved to data/processed/images/")
    print("Features saved to data/processed/kinematic_features_master.csv")

if __name__ == "__main__":
    run_batch()