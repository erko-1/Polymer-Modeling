import numpy as np
import scipy.optimize as opt

# Parameter aus dem Code
k_FENE = 40.0
R_0 = 2.5
sigma = 1.0
epsilon = 1.0

# FENE-Potential
def fene_potential(r):
    if r >= R_0:
        return np.inf  # Verhindert Überschreitung von R_0
    return -0.5 * k_FENE * R_0**2 * np.log(1 - (r / R_0) ** 2)

# Lennard-Jones (WCA) Potential
def wca_potential(r):
    if r >= 2**(1/6) * sigma:
        return 0
    sr6 = (sigma / r)**6
    sr12 = sr6**2
    return 4 * epsilon * (sr12 - sr6) + epsilon

# Gesamtpotential
def total_potential(r):
    return fene_potential(r) + wca_potential(r)

# Minimum numerisch bestimmen
r_min = opt.minimize_scalar(total_potential, bounds=(0.5, R_0), method='bounded')
r_min_value = r_min.x
r_min_value
print(r_min_value)