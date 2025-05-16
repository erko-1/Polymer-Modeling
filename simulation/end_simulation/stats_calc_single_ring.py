import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import os

# === Teil 1: Rg_squared pro Datei auswerten ===
rg_files = glob.glob("SR_Rg_squared*.txt")

# Zielordner für Histogramme
output_dir = "Rg_histograms_sr"
os.makedirs(output_dir, exist_ok=True)

for file in rg_files:
    try:
        df = pd.read_csv(file, sep="\t", comment="#")
        if 'Rg_squared' in df.columns:
            rg_values = df['Rg_squared'].values

            # Plot Histogramm für diese Datei
            plt.figure(figsize=(8, 4))
            plt.hist(rg_values, bins=30, color='blue', alpha=0.7, edgecolor='black')
            plt.title(fr"Histogramm von $R_g^2$ (SR)\nDatei: {os.path.basename(file)}")
            plt.xlabel(r"$R_g^2$")
            plt.ylabel("Häufigkeit")
            plt.grid(True)
            plt.tight_layout()

            # Speichern der Abbildung
            filename = os.path.splitext(os.path.basename(file))[0] + "_histogram.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

        else:
            print(f"Datei übersprungen (fehlende Spalte): {file}")
    except Exception as e:
        print(f"Fehler beim Einlesen von {file}: {e}")


# === Teil 2: Eigenwertanalyse wie bei DR, aber für 1 Ring ===

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

ev_files = glob.glob("SR_Eigenvalues*.txt")

for file in ev_files:
    try:
        df = pd.read_csv(file, sep="\t")
        required = {'Step', 'Eig_val_1', 'Eig_val_2', 'Eig_val_3'}
        if not required.issubset(df.columns):
            print(f"Datei übersprungen (fehlende Spalten): {file}")
            continue

        df['Step'] = df['Step'].apply(lambda s: int(str(s).split('-')[0]))

        # Sortierte Eigenwerte berechnen
        λ1, λ2, λ3 = sort_eigenvalues(df['Eig_val_1'], df['Eig_val_2'], df['Eig_val_3'])
        mean_vals = (np.mean(λ1), np.mean(λ2), np.mean(λ3))

        # Asphärizität & Anisotropie
        b_vals, kappa2_vals = compute_anisotropy(df['Eig_val_1'], df['Eig_val_2'], df['Eig_val_3'])
        b_mean, b_std = np.mean(b_vals), np.std(b_vals)
        kappa2_mean, kappa2_std = np.mean(kappa2_vals), np.std(kappa2_vals)

        # Ausgabe schreiben
        base = os.path.splitext(os.path.basename(file))[0]
        out_file = f"{base}_Analyse.txt"

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("Mittlere sortierte Eigenwerte:\n")
            f.write(f"{'λ1':>10} | {'λ2':>10} | {'λ3':>10}\n")
            f.write("-" * 34 + "\n")
            f.write(f"{mean_vals[0]:10.4f} | {mean_vals[1]:10.4f} | {mean_vals[2]:10.4f}\n\n")

            f.write("Asphärizität und Anisotropie:\n")
            f.write(f"Asphärizität b: {b_mean:.6f} ± {b_std:.6f}\n")
            f.write(f"Anisotropie κ²: {kappa2_mean:.6f} ± {kappa2_std:.6f}\n")

        print(f"Analyse abgeschlossen: {out_file}")

    except Exception as e:
        print(f"Fehler bei Datei {file}: {e}")
