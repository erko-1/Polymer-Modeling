import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os

k_B = 1.0 
T = 1.0 
beta = 1 / (k_B * T)

def compute_slope(filename):
    try:
        angles = np.loadtxt(filename, usecols=1, skiprows=500)

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
file_list = glob.glob("DR_angles*.txt")

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

# === Teil 1: Rg_squared auswerten ===
rg_files = glob.glob("DR_Rg_squared_values*.txt")

output_dir = "Rg_histograms_dr"
os.makedirs(output_dir, exist_ok=True)

for file in rg_files:
    try:
        df = pd.read_csv(file, sep="\t", comment="#")
        required = {'Step_Range', 'Rg_squared_ring1', 'Rg_squared_ring2', 'Rg_squared_total'}
        if not required.issubset(df.columns):
            print(f"Datei übersprungen (fehlende Spalten): {file}")
            continue

        base_name = os.path.splitext(os.path.basename(file))[0]

        # Histogramm für Ring 1
        plt.figure(figsize=(8, 4))
        plt.hist(df['Rg_squared_ring1'], bins=30, color='blue', alpha=0.7, edgecolor='black')
        plt.title(rf"Histogramm von $R_g^2$ (Ring 1)\nDatei: {base_name}")
        plt.xlabel(r"$R_g^2$")
        plt.ylabel("Häufigkeit")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_Ring1.png"))
        plt.close()

        # Histogramm für Ring 2
        plt.figure(figsize=(8, 4))
        plt.hist(df['Rg_squared_ring2'], bins=30, color='green', alpha=0.7, edgecolor='black')
        plt.title(rf"Histogramm von $R_g^2$ (Ring 2)\nDatei: {base_name}")
        plt.xlabel(r"$R_g^2$")
        plt.ylabel("Häufigkeit")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_Ring2.png"))
        plt.close()

        # Histogramm für Gesamt
        plt.figure(figsize=(8, 4))
        plt.hist(df['Rg_squared_total'], bins=30, color='purple', alpha=0.7, edgecolor='black')
        plt.title(rf"Histogramm von $R_g^2$ (Gesamt)\nDatei: {base_name}")
        plt.xlabel(r"$R_g^2$")
        plt.ylabel("Häufigkeit")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_Total.png"))
        plt.close()

        print(f"Histogramme gespeichert für Datei: {file}")

    except Exception as e:
        print(f"Fehler beim Einlesen von {file}: {e}")


# === Hilfsfunktionen ===
def sort_eigenvalues(ev1, ev2, ev3):
    eigenvalues = np.stack([ev1, ev2, ev3], axis=1)
    sorted_vals = np.sort(eigenvalues, axis=1)[:, ::-1]  # descending
    return sorted_vals[:, 0], sorted_vals[:, 1], sorted_vals[:, 2]

def compute_anisotropy(ev1, ev2, ev3):
    λ1, λ2, λ3 = sort_eigenvalues(ev1, ev2, ev3)
    λs = np.stack([λ1, λ2, λ3], axis=1)
    mean_λ = np.mean(λs, axis=1)

    b = λ1 - 0.5 * (λ2 + λ3)
    kappa2 = ((λ1 - mean_λ)**2 + (λ2 - mean_λ)**2 + (λ3 - mean_λ)**2) / (2 * (np.sum(λs, axis=1))**2)
    return b, kappa2

# === Daten einlesen und je Datei analysieren ===
ev_files = glob.glob("DR_Eigenvalues*.txt")

for file in ev_files:
    try:
        df = pd.read_csv(file, sep='\t')
        required = {'Step', 'EV_Ring1_1', 'EV_Ring1_2', 'EV_Ring1_3',
                    'EV_Ring2_1', 'EV_Ring2_2', 'EV_Ring2_3',
                    'EV_Total_1', 'EV_Total_2', 'EV_Total_3'}
        if not required.issubset(df.columns):
            print(f"Datei übersprungen (fehlende Spalten): {file}")
            continue

        df['Step'] = df['Step'].apply(lambda s: int(s.split('-')[0]) if isinstance(s, str) else s)

        # Ring 1
        λ1_r1, λ2_r1, λ3_r1 = sort_eigenvalues(df['EV_Ring1_1'], df['EV_Ring1_2'], df['EV_Ring1_3'])
        mean_r1 = (np.mean(λ1_r1), np.mean(λ2_r1), np.mean(λ3_r1))

        # Ring 2
        λ1_r2, λ2_r2, λ3_r2 = sort_eigenvalues(df['EV_Ring2_1'], df['EV_Ring2_2'], df['EV_Ring2_3'])
        mean_r2 = (np.mean(λ1_r2), np.mean(λ2_r2), np.mean(λ3_r2))

        # Total
        λ1_tot, λ2_tot, λ3_tot = sort_eigenvalues(df['EV_Total_1'], df['EV_Total_2'], df['EV_Total_3'])
        mean_tot = (np.mean(λ1_tot), np.mean(λ2_tot), np.mean(λ3_tot))

        # Asphärizität & Anisotropie für Gesamtstruktur
        b_vals, kappa2_vals = compute_anisotropy(df['EV_Total_1'], df['EV_Total_2'], df['EV_Total_3'])
        b_mean, b_std = np.mean(b_vals), np.std(b_vals)
        kappa2_mean, kappa2_std = np.mean(kappa2_vals), np.std(kappa2_vals)

        # === Dateiname für Ausgabe
        base = os.path.splitext(os.path.basename(file))[0]
        out_file = f"{base}_Analyse.txt"

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("Vergleich der mittleren sortierten Eigenwerte:\n")
            f.write(f"{'System':<10} | {'λ1':>10} | {'λ2':>10} | {'λ3':>10}\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Ring 1':<10} | {mean_r1[0]:10.4f} | {mean_r1[1]:10.4f} | {mean_r1[2]:10.4f}\n")
            f.write(f"{'Ring 2':<10} | {mean_r2[0]:10.4f} | {mean_r2[1]:10.4f} | {mean_r2[2]:10.4f}\n")
            f.write(f"{'Total':<10} | {mean_tot[0]:10.4f} | {mean_tot[1]:10.4f} | {mean_tot[2]:10.4f}\n\n")
            f.write("Asphärizität und Anisotropie (nur Gesamt):\n")
            f.write(f"Asphärizität b: {b_mean:.6f} ± {b_std:.6f}\n")
            f.write(f"Anisotropie κ²: {kappa2_mean:.6f} ± {kappa2_std:.6f}\n")

        print(f"Analyse abgeschlossen: {out_file}")

    except Exception as e:
        print(f"Fehler bei Datei {file}: {e}")
