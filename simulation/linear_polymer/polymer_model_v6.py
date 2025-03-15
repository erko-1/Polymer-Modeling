import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Simulation parameters
num_monomers = 20  # Number of monomers in the polymer
box_size = 10.0  # Size of simulation box
k_FENE = 40.0  # FENE bond strength
R_0 = 2.5  # FENE bond maximum extension
sigma = 1.0  # Lennard-Jones sigma
epsilon = 1.0  # Lennard-Jones epsilon
T = 1.0  # Temperature in reduced units
k_B = 1.0  # Boltzmann constant
num_steps = 100000  # Number of Monte Carlo steps
step_size = 0.1  # Maximum movement per step

# Initialize polymer as a linear chain
polymer = np.cumsum(np.random.randn(num_monomers, 2) * 0.1, axis=0)

# FENE potential function
def fene_potential(r):
    if r >= R_0:
        return np.inf  # Prevent bond extension beyond R_0
    return -0.5 * k_FENE * R_0**2 * np.log(1 - (r / R_0) ** 2)

# Lennard-Jones potential function
def lj_potential(r):
    if r == 0:
        return np.inf
    if r < 2**(1/6) * sigma:  #cutoff
        sr6 = (sigma / r)**6
        sr12 = sr6**2
        return 4 * epsilon * (sr12 - sr6) + epsilon
    if r >= 2**(1/6) * sigma:
        return 0

# Total energy calculation
def total_energy(polymer):
    energy = 0.0
    distances = squareform(pdist(polymer))
    
    # Bonded interactions (FENE)
    for i in range(num_monomers - 1):
        energy += fene_potential(distances[i, i + 1])
    
    # Non-bonded interactions (Lennard-Jones)
    for i in range(num_monomers):
        for j in range(i + 2, num_monomers):  # Avoid bonded pairs
            energy += lj_potential(distances[i, j])
    
    return energy

# Monte Carlo simulation
energies = []
for step in range(num_steps):
    i = np.random.randint(num_monomers)  # Select a random monomer
    displacement = (np.random.rand(2) - 0.5) * step_size  # Random move
    new_polymer = polymer.copy()
    new_polymer[i] += displacement
    
    # Apply Metropolis criterion
    dE = total_energy(new_polymer) - total_energy(polymer)
    if dE < 0 or np.random.rand() < np.exp(-dE / (k_B*T)):
        polymer = new_polymer
    
    # Store energy for analysis
    energies.append(total_energy(polymer))

# Plot energy over time
#plt.plot(energies)
#plt.xlabel('MC Steps')
#plt.ylabel('Total Energy')
#plt.title('Energy Convergence in Monte Carlo Simulation')
#plt.show()

# 3D Visualization of Polymer Chain
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(polymer[:, 0], polymer[:, 1], marker='o', linestyle='-')
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.set_zlabel('Z')
ax.set_title('Polymer Chain Configuration')
plt.show()

