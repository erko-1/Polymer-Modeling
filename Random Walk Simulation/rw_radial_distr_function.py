import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Function to initialize particle positions
def initialize_particles(num_particles, box_size):
    return np.random.uniform(0, box_size, (num_particles, 2))  # 2D positions

# Function to apply periodic boundary conditions
def apply_periodic_boundary(position, box_size):
    return position % box_size

# Function to calculate Lennard-Jones potential
def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    if r == 0:
        return np.inf
    if r < 2**(1/6) * sigma:
        sr6 = (sigma / r)**6
        sr12 = sr6**2
        return 4 * epsilon * (sr12 - sr6) + epsilon
    if r >= 2**(1/6) * sigma:
        return 0

# Optimized function for total potential energy using a cell list
def total_potential_energy_optimized(positions, box_size, epsilon=1.0, sigma=1.0, cell_size=None):
    if cell_size is None:
        cell_size = sigma 
    
    # Create a grid of cells
    num_cells = int(np.ceil(box_size / cell_size))
    cells = [[] for _ in range(num_cells**2)]
    
    # Assign particles to cells
    for idx, position in enumerate(positions):
        cell_idx = (position // cell_size).astype(int)
        cell_index_1d = cell_idx[0] * num_cells + cell_idx[1]
        cells[cell_index_1d].append(idx)
    
    # Compute energy
    energy = 0.0
    for c_idx, cell in enumerate(cells):
        if not cell:  # Skip empty cells
            continue
        # Neighboring cells (including self)
        neighbors = []
        x, y = divmod(c_idx, num_cells)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = (x + dx) % num_cells, (y + dy) % num_cells
                neighbors.append(nx * num_cells + ny)
        
        # Compute pairwise interactions within the cell and neighboring cells
        for i in cell:
            for neighbor in neighbors:
                for j in cells[neighbor]:
                    if i >= j:
                        continue  # Avoid double counting
                    delta = positions[i] - positions[j]
                    delta -= box_size * np.round(delta / box_size)  # Periodic boundary conditions
                    r = np.linalg.norm(delta)
                    energy += lennard_jones_potential(r, epsilon, sigma)
    return energy

# Optimized Monte Carlo step using cell lists
def monte_carlo_step_optimized(positions, step_size, box_size, epsilon=1.0, sigma=1.0, temperature=1.0, boltzmann_const=1.0, cell_size=None):
    new_positions = positions.copy()
    cell_size = sigma 
    
    for i in range(positions.shape[0]):
        # Random step for a single particle
        step = np.random.uniform(-step_size, step_size, positions.shape[1])
        trial_position = apply_periodic_boundary(new_positions[i] + step, box_size)
        
        # Compute energy difference efficiently
        original_position = new_positions[i].copy()
        new_positions[i] = trial_position
        energy_new = total_potential_energy_optimized(new_positions, box_size, epsilon, sigma, cell_size)
        new_positions[i] = original_position
        energy_old = total_potential_energy_optimized(new_positions, box_size, epsilon, sigma, cell_size)
        
        delta_energy = energy_new - energy_old
        
        # Accept or reject move based on Boltzmann criterion
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / (boltzmann_const * temperature)):
            new_positions[i] = trial_position  # Accept move

            
    return new_positions

# Main function to simulate the random walk
def random_walk_simulation_optimized(num_particles=100, box_size=10.0, step_size=1.0, num_steps=100, epsilon=1.0, sigma=1.0, temperature=1.0):
    # Initialize particle positions
    positions = initialize_particles(num_particles, box_size)
    trajectory = [positions.copy()]
    cell_size = sigma
    
    for _ in range(num_steps):
        positions = monte_carlo_step_optimized(positions, step_size, box_size, epsilon, sigma, temperature, cell_size=cell_size)
        trajectory.append(positions.copy())
    
    return trajectory

# Compute the distance between all pairs of particles in the system
def compute_distance(positions, box_length):
    n_atoms = len(positions)
    distances = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i >= j:
                continue
            dr = positions[i] - positions[j]
            dr = dr - box_length * np.round(dr / box_length)
            distance = np.linalg.norm(dr)
            distances.append(distance)
    return distances

# Bin the distances into radial bins
def bin_distances(distances, bin_width, max_distance):
    bins = np.arange(bin_width, max_distance + bin_width * 2, bin_width)
    hist, _ = np.histogram(distances, bins=bins)
    return hist

def radial_distribution_function(hist, n_atoms, box_length, bin_width):
    rho = n_atoms / box_length**2  # 2D density
    r = (np.arange(1, len(hist) + 1) - 0.5) * bin_width
    shell_areas = 2 * np.pi * r * bin_width
    g = hist / (rho * n_atoms * shell_areas)
    return r, g

# Parameters
num_particles = 100     # Number of particles
box_size = 10.0          # Size of the box
step_size = 0.1          # Step size for the random walk
num_steps = 100          # Number of steps to simulate
epsilon = 1.0            # Epsilon 
sigma = 1.0              # Sigma - distance where potential is zero
temperature = 0.05        # Simulation temperature
bin_width = 0.05          # Bin width for RDF calculation

# Run the simulation
trajectory = random_walk_simulation_optimized(num_particles, box_size, step_size, num_steps, epsilon, sigma, temperature)

# Use the last step of the trajectory for RDF calculation
final_positions = trajectory[-1]

# Compute distances and RDF
distances = compute_distance(final_positions, box_size)
hist = bin_distances(distances, bin_width, box_size / 2)
r, g = radial_distribution_function(hist, num_particles, box_size, bin_width)

# Plot the RDF
plt.figure(figsize=(8, 6))
plt.plot(r, g, label=f"Radial Distribution Function for N = {num_particles} and T = {temperature}")
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Radial Distribution Function")
plt.legend()
plt.grid()
plt.show()
