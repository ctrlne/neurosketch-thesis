import pandas as pd
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

import matplotlib.pyplot as plt
import os

class NeuroSketchPreprocessor:
    def __init__(self, target_hz=133, img_size=(224, 224)):
        """
        Initializes the preprocessor to match PaHaW standards.
        - target_hz: 133 Hz (Standard baseline sampling rate)
        - img_size: 224x224 (Standard for ResNet-18 & VGG-16)
        """
        self.target_hz = target_hz
        self.img_size = img_size
        
    # STREAM A: KINEMATIC (NUMBERS -> Random Forest)
    def process_kinematics(self, df):
        # Convert columns to flat NumPy arrays to prevent index errors
        time_arr = df['time'].to_numpy()
        x_smooth = gaussian_filter1d(df['x'].to_numpy(), sigma=2)
        y_smooth = gaussian_filter1d(df['y'].to_numpy(), sigma=2)
        pressure_smooth = gaussian_filter1d(df['pressure'].to_numpy(), sigma=2)
        
        # Derive kinematics
        vel_x = np.gradient(x_smooth, time_arr)
        vel_y = np.gradient(y_smooth, time_arr)
        velocity = np.sqrt(vel_x**2 + vel_y**2)
        
        accel = np.gradient(velocity, time_arr)
        jerk = np.gradient(accel, time_arr)
        
        # 4-6 Hz Tremor extraction
        freqs, psd = welch(velocity, fs=self.target_hz, nperseg=min(256, len(df)))
        tremor_band = (freqs >= 4) & (freqs <= 6)
        tremor_power_4_to_6 = np.trapezoid(psd[tremor_band], freqs[tremor_band])
        
        return {
            'mean_velocity': velocity.mean(),
            'pressure_variance': pressure_smooth.var(),
            'jerk_variance': jerk.var(),
            'tremor_power_4_to_6_hz': tremor_power_4_to_6
        }

    # STREAM B: VISUAL (IMAGES -> ResNet-18 CNN)
    def process_visuals(self, image_path):
        # 1. Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. Resize to 224x224
        img_resized = cv2.resize(img, self.img_size)
        
        # 3. Canny Edge Detection 
        edges = cv2.Canny(img_resized, threshold1=100, threshold2=200)
        
        return edges



if __name__ == "__main__":
    processor = NeuroSketchPreprocessor()
    
    # 1. Update this path to point to an .svc file!
    sample_svc_path = r"data\raw\data\PaHaW\PaHaW_public\00001\00001__1_1.svc"
    
    print("--- TESTING KINEMATIC STREAM ---")
    try:
        # Read the .svc based on info.txt (skip row 1, separate by spaces)
        df = pd.read_csv(sample_svc_path, sep=r'\s+', skiprows=1, 
                        names=['y', 'x', 'time', 'button', 'azimuth', 'altitude', 'pressure'])
        
        kinematic_results = processor.process_kinematics(df)
        print("Kinematic Extraction Success! Biomarkers:")
        print(kinematic_results)
    except Exception as e:
        print(f"Kinematic Error: {e}")

    print("\n--- RECONSTRUCTING VISUAL IMAGE ---")
    try:
        # Filter to only show when the pen was actually touching the tablet (button == 1)
        df_pen_down = df[df['button'] == 1]
        
        # Plot the X and Y coordinates to recreate the drawing
        plt.figure(figsize=(5,5), facecolor='white')
        plt.plot(df_pen_down['x'], -df_pen_down['y'], color='black', linewidth=2) # -y to flip it upright
        plt.axis('off')
        
        # Save the rendered drawing
        generated_img_path = r"data\processed\generated_spiral.png"
        plt.savefig(generated_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Successfully generated drawing at {generated_img_path}")
        
        # Now pass this generated image into our Canny CNN processor!
        edges = processor.process_visuals(generated_img_path)
        cv2.imwrite(r"data\processed\final_cnn_input.png", edges)
        print("Successfully applied Canny Edge Detection for ResNet-18!")
        
    except Exception as e:
        print(f"Visual Error: {e}")