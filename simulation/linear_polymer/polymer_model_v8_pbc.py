import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform

# Simulation parameters
num_monomers = 20
box_size = 10.0
k_FENE = 40.0
sigma = 1.0
epsilon = 1.0
R_0 = 2.5 * sigma
T = 1.0
k_B = 1.0
num_steps = 100000
step_size = 0.1
optimal_bond_length = 0.9651375794516794

# Initialize polymer inside the box
polymer = np.array([[i * optimal_bond_length % box_size, 0, 0] for i in range(num_monomers)])

# Function to apply periodic boundary conditions (PBC)
def apply_pbc(position, box_size):
    return position % box_size

# Function for Minimum Image Convention (MIC)
def minimum_image_distance(pos1, pos2, box_size):
    delta = pos1 - pos2
    return delta - box_size * np.round(delta / box_size)

# Potential functions
def fene_potential(r):
    if r >= R_0:
        return np.inf
    return -0.5 * k_FENE * R_0**2 * np.log(1 - (r / R_0) ** 2)

def lj_potential(r):
    if r == 0:
        return np.inf
    if r < 2**(1/6) * sigma:
        sr6 = (sigma / r)**6
        sr12 = sr6**2
        return 4 * epsilon * (sr12 - sr6) + epsilon
    return 0

# Function to calculate total energy with PBC
def total_energy(polymer):
    energy = 0.0
    num_monomers = len(polymer)
    distances = np.zeros((num_monomers, num_monomers))

    # Compute distances using Minimum Image Convention (MIC)
    for i in range(num_monomers):
        for j in range(i + 1, num_monomers):
            r_ij = np.linalg.norm(minimum_image_distance(polymer[i], polymer[j], box_size))
            distances[i, j] = r_ij
            distances[j, i] = r_ij  # Symmetric matrix

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
    i = np.random.randint(num_monomers)
    displacement = (np.random.rand(3) - 0.5) * step_size  
    new_polymer = polymer.copy()
    new_polymer[i] += displacement
    
    # Apply periodic boundary conditions (PBC)
    new_polymer[i] = apply_pbc(new_polymer[i], box_size)

    # Apply Metropolis criterion
    dE = total_energy(new_polymer) - total_energy(polymer)
    if dE < 0 or np.random.rand() < np.exp(-dE / (k_B*T)):
        polymer = new_polymer
    
    # Store energy for analysis
    energies.append(total_energy(polymer))

# 3D Visualization of Polymer Chain
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(polymer[:, 0], polymer[:, 1], polymer[:, 2], marker='o', linestyle='-')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Polymer Chain Configuration (3D) with PBC')
plt.show()

# Plot energy convergence
plt.plot(energies)
plt.xlabel('MC Steps')
plt.ylabel('Total Energy')
plt.title('Energy Convergence in Monte Carlo Simulation with PBC')
plt.show()
