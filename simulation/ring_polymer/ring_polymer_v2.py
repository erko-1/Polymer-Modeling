import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulation parameters
num_monomers = 10  # Number of monomers in the polymer
box_size = 10.0  # Size of simulation box (cubic)
k_FENE = 40.0  # FENE bond strength
sigma = 1.0  # Lennard-Jones sigma
epsilon = 1.0  # Lennard-Jones epsilon
R_0 = 2.5 * sigma  # FENE bond maximum extension
T = 1.0  # Temperature in reduced units
k_B = 1.0  # Boltzmann constant
num_steps = 10000  # Number of Monte Carlo steps
step_size = 0.1  # Maximum movement per step
optimal_bond_length = 0.9651375794516794  # Precomputed minimum distance

# Radius for the ring configuration
radius = (optimal_bond_length * num_monomers) / (2 * np.pi)
theta = optimal_bond_length / radius  # Angle between consecutive monomers
angles = np.arange(0, num_monomers) * theta

# Initialize polymer positions (ring configuration)
polymer = np.array([
    np.cos(angles) * radius,  # X-coordinates
    np.sin(angles) * radius,  # Y-coordinates
    np.zeros(num_monomers)    # Z-coordinates 
]).T

def apply_pbc(position, box_size):
    """Apply periodic boundary conditions."""
    return position - box_size * np.round(position / box_size)

def minimum_image_distance(r1, r2, box_size):
    """Apply Minimum Image Convention for periodic boundary conditions."""
    delta = r1 - r2
    return delta - box_size * np.round(delta / box_size)

def fene_potential(r):
    """FENE potential."""
    if r >= R_0:
        return np.inf  
    return -0.5 * k_FENE * R_0**2 * np.log(1 - (r / R_0) ** 2)

def lj_potential(r):
    """Lennard-Jones (WCA) potential."""
    if r == 0:
        return np.inf
    if r < 2**(1/6) * sigma:  # Cutoff
        sr6 = (sigma / r)**6
        sr12 = sr6**2
        return 4 * epsilon * (sr12 - sr6) + epsilon
    return 0

def total_energy(polymer):
    """Calculate total energy of the polymer considering FENE and Lennard-Jones potentials."""
    energy = 0.0
    
    # Bonded interactions (FENE) between consecutive monomers
    for i in range(num_monomers):
        j = (i + 1) % num_monomers  # Connect the last monomer to the first (ring closure)
        r_ij = minimum_image_distance(polymer[i], polymer[j], box_size)
        distance = np.linalg.norm(r_ij)
        energy += fene_potential(distance)
    
    # Non-bonded interactions (Lennard-Jones) between non-bonded monomers
    for i in range(num_monomers):
        for j in range(i + 2, num_monomers):  # Avoid bonded pairs
            r_ij = minimum_image_distance(polymer[i], polymer[j], box_size)
            distance = np.linalg.norm(r_ij)
            energy += lj_potential(distance)
    
    return energy

def unwrap_polymer(polymer, box_size):
    """Unwrap polymer coordinates for visualization so that it appears continuous."""
    unwrapped_polymer = polymer.copy()
    
    # Unwrap positions based on PBC
    for i in range(1, num_monomers):
        delta = polymer[i] - polymer[i - 1]
        delta -= box_size * np.round(delta / box_size)  # Apply minimum image convention
        unwrapped_polymer[i] = unwrapped_polymer[i - 1] + delta  # Extend unwrapped coordinates
    
    # Ensure the last monomer is connected back to the first for continuous visualization
    delta = polymer[0] - polymer[-1]
    delta -= box_size * np.round(delta / box_size)
    unwrapped_polymer[-1] = unwrapped_polymer[0] + delta  

    return unwrapped_polymer

# Monte Carlo simulation with PBC
energies = []
for step in range(num_steps):
    i = np.random.randint(num_monomers)
    displacement = (np.random.rand(3) - 0.5) * step_size
    new_polymer = polymer.copy()
    new_polymer[i] += displacement
    new_polymer[i] = apply_pbc(new_polymer[i], box_size)  # Apply PBC
    
    # Apply Boltzmann criterion
    dE = total_energy(new_polymer) - total_energy(polymer)
    if dE < 0 or np.random.rand() < np.exp(-dE / (k_B * T)):
        polymer = new_polymer

    energies.append(total_energy(polymer))  # Store energy for analysis

# Unwrap polymer for visualization
unwrapped_polymer = unwrap_polymer(polymer, box_size)

# 3D Visualization of the Ring Polymer with Correct PBC
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(unwrapped_polymer[:, 0], unwrapped_polymer[:, 1], unwrapped_polymer[:, 2], marker='o', linestyle='-')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ring Polymer Configuration with PBC Visualization')
plt.show()

# Plot energy convergence
plt.plot(energies)
plt.xlabel('MC Steps')
plt.ylabel('Total Energy')
plt.title('Energy Convergence in Monte Carlo Simulation')
plt.show()
