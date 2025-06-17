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
        angles = np.loadtxt(filename, usecols=1, skiprows=2000)

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
first_file = file_list[4]
angles = np.loadtxt(first_file, usecols=1, skiprows=400)
num_bins = 40
plt.figure(figsize=(8, 5))
plt.hist(angles, bins=num_bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Histogram of Angles 4")
plt.xlabel("Angle (radians)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------------------

output_dir = "Rg_histograms_dr"
os.makedirs(output_dir, exist_ok=True)

rg_files = glob.glob("DR_Rg_squared_values*.txt")

# Listen für das Sammeln aller Daten über alle Dateien
all_ring1 = []
all_ring2 = []
all_total = []

mean_values = []  # Liste für mittlere Werte pro Datei (zur späteren Mittelung)

# === Daten aus allen Dateien sammeln ===
for file in rg_files:
    try:
        df = pd.read_csv(file, sep="\t", comment="#")
        required = {'Step_Range', 'Rg_squared_ring1', 'Rg_squared_ring2', 'Rg_squared_total'}
        if not required.issubset(df.columns):
            print(f"Datei übersprungen (fehlende Spalten): {file}")
            continue

        all_ring1.extend(df['Rg_squared_ring1'])
        all_ring2.extend(df['Rg_squared_ring2'])
        all_total.extend(df['Rg_squared_total'])

        # Mittelwerte pro Datei berechnen
        mean_values.append({
            "file": os.path.basename(file),
            "mean_ring1": df['Rg_squared_ring1'].mean(),
            "mean_ring2": df['Rg_squared_ring2'].mean(),
            "mean_total": df['Rg_squared_total'].mean(),
        })

    except Exception as e:
        print(f"Fehler beim Einlesen von {file}: {e}")

# === Histogramme über alle Dateien hinweg erstellen ===
def plot_hist(data, title, filename, color):
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=30, color=color, alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel(r"$R_g^2$")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_hist(all_ring1, r"Histogram of $R_g^2$ (Ring 1)", "Ring1_combined.png", "blue")
plot_hist(all_ring2, r"Histogram of $R_g^2$ (Ring 2)", "Ring2_combined.png", "blue")
plot_hist(all_total, r"Histogram of $R_g^2$ (Total)", "Total_combined.png", "blue")

# === Gemittelte Rg-Werte speichern ===
mean_ring1_all = np.mean([entry["mean_ring1"] for entry in mean_values])
mean_ring2_all = np.mean([entry["mean_ring2"] for entry in mean_values])
mean_total_all = np.mean([entry["mean_total"] for entry in mean_values])

with open(os.path.join(output_dir, "mean_rg_values.txt"), "w") as f:
    f.write("Gemittelte Rg^2-Werte über alle Dateien:\n")
    f.write(f"Ring 1: {mean_ring1_all:.5f}\n")
    f.write(f"Ring 2: {mean_ring2_all:.5f}\n")
    f.write(f"Gesamtsystem: {mean_total_all:.5f}\n")

print("✅ Kombinierte Histogramme und Mittelwerte wurden erstellt.")


# === Hilfsfunktionen ===
def sort_eigenvalues(ev1, ev2, ev3):
    eigenvalues = np.stack([ev1, ev2, ev3], axis=1)
    sorted_vals = np.sort(eigenvalues, axis=1)[:, ::-1]  # descending
    return sorted_vals[:, 0], sorted_vals[:, 1], sorted_vals[:, 2]

def compute_anisotropy(ev1, ev2, ev3):
    λ1, λ2, λ3 = sort_eigenvalues(ev1, ev2, ev3)
    I = λ1 * λ2 + λ2 * λ3 + λ3 * λ1
    Rg2 = λ1 + λ2 + λ3
    b = 1 - 3 * I / (Rg2 ** 2)
    return b

def compute_prolateness(ev1, ev2, ev3):
    λ1, λ2, λ3 = sort_eigenvalues(ev1, ev2, ev3)
    Rg2 = λ1 + λ2 + λ3
    S = (3 * λ1 - Rg2) * (3 * λ2 - Rg2) * (3 * λ3 - Rg2) / (Rg2 ** 3)
    return S

# === Daten einlesen ===
ev_files = glob.glob("DR_Eigenvalues*.txt")
output_dir = "EV_analysis_dr"
os.makedirs(output_dir, exist_ok=True)

# Container für globale Mittelwerte
all_ev_ring1 = []
all_ev_ring2 = []
all_ev_total = []

# === Pro Datei analysieren ===
for file in ev_files:
    try:
        df = pd.read_csv(file, sep='\t')
        required = {'Step', 'EV_Ring1_1', 'EV_Ring1_2', 'EV_Ring1_3',
                    'EV_Ring2_1', 'EV_Ring2_2', 'EV_Ring2_3',
                    'EV_Total_1', 'EV_Total_2', 'EV_Total_3'}
        if not required.issubset(df.columns):
            print(f"Datei übersprungen: {file}")
            continue

        # Zeit als Ganzzahl
        df['Step'] = df['Step'].apply(lambda s: int(s.split('-')[0]) if isinstance(s, str) else s)

        # Eigenwerte sortieren
        λ1_r1, λ2_r1, λ3_r1 = sort_eigenvalues(df['EV_Ring1_1'], df['EV_Ring1_2'], df['EV_Ring1_3'])
        λ1_r2, λ2_r2, λ3_r2 = sort_eigenvalues(df['EV_Ring2_1'], df['EV_Ring2_2'], df['EV_Ring2_3'])
        λ1_tot, λ2_tot, λ3_tot = sort_eigenvalues(df['EV_Total_1'], df['EV_Total_2'], df['EV_Total_3'])

        all_ev_ring1.extend(np.stack([λ1_r1, λ2_r1, λ3_r1], axis=1))
        all_ev_ring2.extend(np.stack([λ1_r2, λ2_r2, λ3_r2], axis=1))
        all_ev_total.extend(np.stack([λ1_tot, λ2_tot, λ3_tot], axis=1))

        # Anisotropie und Prolateness berechnen
        b_vals = compute_anisotropy(df['EV_Total_1'], df['EV_Total_2'], df['EV_Total_3'])
        S_vals = compute_prolateness(df['EV_Total_1'], df['EV_Total_2'], df['EV_Total_3'])

        # Rolling Average
        window = 10
        b_smooth = pd.Series(b_vals).rolling(window=window, center=True).mean()
        S_smooth = pd.Series(S_vals).rolling(window=window, center=True).mean()

        # Liniendiagramme
        base = os.path.splitext(os.path.basename(file))[0]

        plt.figure(figsize=(10, 4))
        plt.plot(df['Step'], b_vals, alpha=0.3, label="b", color='blue')
        plt.plot(df['Step'], b_smooth, label=f"b (rolling, w={window})", color='blue')
        plt.title(f"Anisotropie b – {base}")
        plt.xlabel("Step")
        plt.ylabel("Anisotropie b")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base}_Anisotropie.png"))
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.plot(df['Step'], S_vals, alpha=0.3, label="S", color='green')
        plt.plot(df['Step'], S_smooth, label=f"S (rolling, w={window})", color='green')
        plt.title(f"Prolateness S – {base}")
        plt.xlabel("Step")
        plt.ylabel("Prolateness S")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base}_Prolateness.png"))
        plt.close()

        print(f"Analyse abgeschlossen: {base}")

    except Exception as e:
        print(f"Fehler bei Datei {file}: {e}")

# === Mittlere Eigenwerte über alle Dateien ===
all_ev_ring1 = np.array(all_ev_ring1)
all_ev_ring2 = np.array(all_ev_ring2)
all_ev_total = np.array(all_ev_total)

mean_ring1 = np.mean(all_ev_ring1, axis=0)
mean_ring2 = np.mean(all_ev_ring2, axis=0)
mean_total = np.mean(all_ev_total, axis=0)


λ1, λ2, λ3 = mean_total
Rg2 = λ1 + λ2 + λ3
I = λ1*λ2 + λ2*λ3 + λ3*λ1
b_mean = 1 - 3*I / (Rg2**2)
s_mean = ((3*λ1 - Rg2)*(3*λ2 - Rg2)*(3*λ3 - Rg2)) / (Rg2**3)

# === Ausgabe speichern ===
summary_file = os.path.join(output_dir, "gemittelte_Eigenwerte.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("Gemittelte sortierte Eigenwerte (über alle Dateien und Zeitpunkte)\n")
    f.write(f"{'System':<10} | {'λ1':>10} | {'λ2':>10} | {'λ3':>10}\n")
    f.write("-" * 45 + "\n")
    f.write(f"{'Ring 1':<10} | {mean_ring1[0]:10.4f} | {mean_ring1[1]:10.4f} | {mean_ring1[2]:10.4f}\n")
    f.write(f"{'Ring 2':<10} | {mean_ring2[0]:10.4f} | {mean_ring2[1]:10.4f} | {mean_ring2[2]:10.4f}\n")
    f.write(f"{'Total':<10} | {mean_total[0]:10.4f} | {mean_total[1]:10.4f} | {mean_total[2]:10.4f}\n")
    f.write("\n")
    f.write("\n")
    f.write(f"Total - Prolateness_mean: {s_mean}\n")
    f.write(f"Total - Anisotropy_mean: {b_mean}")    
print(f"✅ Mittelwerte gespeichert: {summary_file}")