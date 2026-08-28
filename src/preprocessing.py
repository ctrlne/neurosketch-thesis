import pandas as pd
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

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
        # 1. Gaussian Low-Pass Filter
        df['x_smooth'] = gaussian_filter1d(df['x'], sigma=2)
        df['y_smooth'] = gaussian_filter1d(df['y'], sigma=2)
        df['pressure_smooth'] = gaussian_filter1d(df['pressure'], sigma=2)
        
        # 2. Derive Dynamic Features
        df['vel_x'] = np.gradient(df['x_smooth'], df['time'])
        df['vel_y'] = np.gradient(df['y_smooth'], df['time'])
        df['velocity'] = np.sqrt(df['vel_x']**2 + df['vel_y']**2)
        
        df['accel'] = np.gradient(df['velocity'], df['time'])
        df['jerk'] = np.gradient(df['accel'], df['time'])
        
        # 3. Fast Fourier Transform (FFT) - 4-6 Hz Tremor Band
        freqs, psd = welch(df['velocity'], fs=self.target_hz, nperseg=min(256, len(df)))
        
        tremor_band = (freqs >= 4) & (freqs <= 6)
        tremor_power_4_to_6 = np.trapz(psd[tremor_band], freqs[tremor_band])
        
        return {
            'mean_velocity': df['velocity'].mean(),
            'pressure_variance': df['pressure_smooth'].var(),
            'jerk_variance': df['jerk'].var(),
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