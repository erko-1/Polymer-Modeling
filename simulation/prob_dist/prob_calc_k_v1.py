import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load angles from multiple runs
angles = np.loadtxt('angles3.txt', usecols=1)  # Assuming angles3.txt stores angles

# Compute histogram (probability distribution)
num_bins = 50
hist, bin_edges = np.histogram(angles, bins=num_bins, density=True)

# Bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Compute cos(alpha)
cos_alpha = np.cos(bin_centers)

# Take logarithm of P(alpha), avoiding log(0) issues
log_P = np.log(hist + 1e-10)  # Small value to prevent log(0)

# Linear fit: log(P) = -K * cos(alpha) + constant
slope, intercept = np.polyfit(cos_alpha, log_P, 1)

# Extract K
K_estimate = -slope

print(f"Estimated K: {K_estimate}")

# Plot
plt.figure()
plt.scatter(cos_alpha, log_P, label='Simulation Data', color='blue')
plt.plot(cos_alpha, slope * cos_alpha + intercept, label=f'Fit (K = {K_estimate:.3f})', color='red')
plt.xlabel('cos(α)')
plt.ylabel('log P(α)')
plt.legend()
plt.title('Estimating K from Monte Carlo Data')
plt.show()
