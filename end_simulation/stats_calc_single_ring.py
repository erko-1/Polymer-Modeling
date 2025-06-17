import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os

# === Teil 1: Rg_squared Histogramme und Mittelwertanalyse ===
rg_files = glob.glob("SR_Rg_squared*.txt")
output_dir = "Rg_histograms_sr"
os.makedirs(output_dir, exist_ok=True)

all_rg_values = []
mean_per_file = []

for file in rg_files:
    try:
        df = pd.read_csv(file, sep="\t", comment="#")
        if 'Rg_squared' in df.columns:
            rg_values = df['Rg_squared'].values
            all_rg_values.extend(rg_values)
            mean_per_file.append(np.mean(rg_values))

            # Einzelnes Histogramm
            plt.figure(figsize=(8, 4))
            plt.hist(rg_values, bins=30, color='blue', alpha=0.7, edgecolor='black')
            plt.title(fr"Histogram of $R_g^2$ (Single Ring)")
            plt.xlabel(r"$R_g^2$")
            plt.ylabel("Probability Distribution")
            plt.grid(True)
            plt.tight_layout()
            plt.close()
        else:
            print(f"Datei übersprungen (fehlende Spalte): {file}")
    except Exception as e:
        print(f"Fehler beim Einlesen von {file}: {e}")

# Kombiniertes Histogramm
plt.figure(figsize=(8, 4))
plt.hist(all_rg_values, bins=30, color='blue', edgecolor='black', alpha=0.8)
plt.title("Histogram of $R_g^2$ (Single Ring)")
plt.xlabel(r"$R_g^2$")
plt.ylabel("Probability Distribution")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Rg_squared_combined.png"))
plt.close()

# Gesamtmittelwert
mean_all = np.mean(all_rg_values)
with open(os.path.join(output_dir, "mean_rg_values.txt"), "w") as f:
    f.write("Gemittelte Rg^2-Werte:\n")
    f.write(f"Über alle Dateien: {mean_all:.5f}\n")

print("✅ Rg^2-Auswertung abgeschlossen.")

# === Teil 2: Eigenwertanalyse & Shape Tensor ===

def sort_eigenvalues(ev1, ev2, ev3):
    eigenvalues = np.stack([ev1, ev2, ev3], axis=1)
    sorted_vals = np.sort(eigenvalues, axis=1)[:, ::-1]
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

ev_files = glob.glob("SR_Eigenvalues*.txt")
output_dir_ev = "EV_analysis_sr"
os.makedirs(output_dir_ev, exist_ok=True)

all_eigenvalues = []

for file in ev_files:
    try:
        df = pd.read_csv(file, sep="\t")
        required = {'Step', 'Eig_val_1', 'Eig_val_2', 'Eig_val_3'}
        if not required.issubset(df.columns):
            print(f"Datei übersprungen (fehlende Spalten): {file}")
            continue

        df['Step'] = df['Step'].apply(lambda s: int(str(s).split('-')[0]))

        λ1, λ2, λ3 = sort_eigenvalues(df['Eig_val_1'], df['Eig_val_2'], df['Eig_val_3'])
        all_eigenvalues.extend(np.stack([λ1, λ2, λ3], axis=1))

        b_vals = compute_anisotropy(df['Eig_val_1'], df['Eig_val_2'], df['Eig_val_3'])
        S_vals = compute_prolateness(df['Eig_val_1'], df['Eig_val_2'], df['Eig_val_3'])

        window = 10
        b_smooth = pd.Series(b_vals).rolling(window=window, center=True).mean()
        S_smooth = pd.Series(S_vals).rolling(window=window, center=True).mean()

        base = os.path.splitext(os.path.basename(file))[0]

        # Plot b
        plt.figure(figsize=(10, 4))
        plt.plot(df['Step'], b_vals, alpha=0.3, label="b", color='blue')
        plt.plot(df['Step'], b_smooth, label=f"b (rolling, w={window})", color='blue')
        plt.title(f"Anisotropie b – {base}")
        plt.xlabel("Step")
        plt.ylabel("Anisotropy b")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_ev, f"{base}_Anisotropy.png"))
        plt.close()

        # Plot S
        plt.figure(figsize=(10, 4))
        plt.plot(df['Step'], S_vals, alpha=0.3, label="S", color='green')
        plt.plot(df['Step'], S_smooth, label=f"S (rolling, w={window})", color='green')
        plt.title(f"Prolateness S – {base}")
        plt.xlabel("Step")
        plt.ylabel("Prolateness S")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_ev, f"{base}_Prolateness.png"))
        plt.close()

        print(f"Analyse abgeschlossen: {base}")

    except Exception as e:
        print(f"Fehler bei Datei {file}: {e}")

# Globale Mittelwerte
all_eigenvalues = np.array(all_eigenvalues)
mean_vals = np.mean(all_eigenvalues, axis=0)
λ1, λ2, λ3 = mean_vals
Rg2 = λ1 + λ2 + λ3
I = λ1 * λ2 + λ2 * λ3 + λ3 * λ1
b_mean = 1 - 3 * I / (Rg2 ** 2)
s_mean = ((3 * λ1 - Rg2) * (3 * λ2 - Rg2) * (3 * λ3 - Rg2)) / (Rg2 ** 3)

summary_file = os.path.join(output_dir_ev, "mean_eigenvalues.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("Gemittelte sortierte Eigenwerte (über alle Dateien)\n")
    f.write(f"{'λ1':>10} | {'λ2':>10} | {'λ3':>10}\n")
    f.write("-" * 34 + "\n")
    f.write(f"{λ1:10.4f} | {λ2:10.4f} | {λ3:10.4f}\n\n")
    f.write(f"Anisotropy b (mean): {b_mean:.6f}\n")
    f.write(f"Prolateness S (mean): {s_mean:.6f}\n")

print(f"✅ Eigenwert-Mittelwerte gespeichert: {summary_file}")
