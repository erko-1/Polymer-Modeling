import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os
from scipy.signal import find_peaks

# Load data
file_list = glob.glob("DR_angles*.txt")
df = pd.read_csv(file_list[1], sep='\s+', header=None, names=['step', 'angle'])

# Histogram to estimate angle peaks
hist, bin_edges = np.histogram(df['angle'], bins=100)
peaks, _ = find_peaks(hist, distance=10)

# Estimate central angles from histogram peaks
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
dominant_angles = bin_centers[peaks]
dominant_angles = sorted(dominant_angles, key=lambda x: -hist[bin_centers.tolist().index(x)])
angle1, angle2 = dominant_angles[:2]  # two most dominant angles

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['angle'], label="Angles of Dataset 2")
plt.axhline(y=angle1, color='red', linestyle='--', label=f'Angle 1 ≈ {angle1:.3f} rad')
plt.axhline(y=angle2, color='green', linestyle='--', label=f'Angle 2 ≈ {angle2:.3f} rad')
plt.xlabel("Step")
plt.ylabel("Angle (rad)")
plt.title("Double Ring - Biangular Structure")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
