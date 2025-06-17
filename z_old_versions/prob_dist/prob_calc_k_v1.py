import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

k_B = 1.0 
T = 1.0 
beta = 1 / (k_B * T)

def compute_slope(filename):
    try:
        angles = np.loadtxt(filename, usecols=1, skiprows=1000)

        num_bins = 50
        hist, bin_edges = np.histogram(angles, bins=num_bins, density=True)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  
        cos_alpha = np.cos(bin_centers)

        log_P = np.log(hist + 1e-10)  # Avoid log(0) errors

        # Linear fit: log(P) = -βK * cos(alpha) + constant
        slope, intercept = np.polyfit(cos_alpha, log_P, 1)

        return slope  # -βK
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

# Get list of files matching "angles*.txt"
file_list = glob.glob("angles*.txt")

slopes = []
K_values = []

with open("K_values.txt", "w", encoding="utf-8") as f:
    f.write("Filename\tSlope (-βK)\t K\n") 
    for file in file_list:
        slope = compute_slope(file)
        if slope is not None:  # Skip files that failed to process
            K = -slope / beta
            slopes.append(slope)
            K_values.append(K)
            f.write(f"{file}\t{slope:.6f}\t {K:.6f}\n")  

if K_values:
    K_mean = np.mean(K_values)
    K_std = np.std(K_values)

    print(f"Estimated K: {K_mean:.3f} ± {K_std:.3f} (for β = {beta})")

    with open("K_values.txt", "a", encoding="utf-8") as f:
        f.write(f"\nMean K: {K_mean:.6f}\n")
        f.write(f"Standard Deviation: {K_std:.6f}\n")

else:
    print("No valid K values computed. Check input files.")



file_list2 = glob.glob("Rg_squared*.txt")
first_file2 = file_list2[0]

df = pd.read_csv(first_file2, delimiter="\t")  # You can also try sep="\s+" if needed

# Extract values
step_labels = df["Step_Range"].values
Rg_values = df["Rg_squared_ring1"].values

# Plot histogram of angles from the first file as an example
first_file = file_list[0]
angles = np.loadtxt(first_file, usecols=1, skiprows=400)
num_bins = 40
plt.figure(figsize=(8, 5))
plt.hist(angles, bins=num_bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.title(f"Histogram of Angles\nFile: {first_file}")
plt.xlabel("Angle (radians)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.tight_layout()
plt.show()


# Plotting
plt.figure(figsize=(15, 5))
plt.bar(step_labels, Rg_values, width=0.8, align="center", color="lightcoral", edgecolor="black")
plt.title(f"Plot of $R_g^2$ of Ring1\nFile: {first_file2}")
plt.xlabel("MC Step Range")
plt.ylabel(r"$R_g^2$")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(True)
plt.show()
