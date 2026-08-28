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
                
                # 3. Generate & Process Image
                df_pen = df[df['button'] == 1]
                img_temp_path = f"{PROCESSED_DIR}/temp.png"
                
                plt.figure(figsize=(5,5), facecolor='white')
                plt.plot(df_pen['x'], -df_pen['y'], color='black', linewidth=2)
                plt.axis('off')
                plt.savefig(img_temp_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                
                edges = processor.process_visuals(img_temp_path)
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