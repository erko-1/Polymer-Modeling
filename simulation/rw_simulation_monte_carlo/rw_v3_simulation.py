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
    if r == 0:  # Prevent division by zero
        return np.inf
    sr6 = (sigma / r)**6
    sr12 = sr6**2
    return 4 * epsilon * (sr12 - sr6)

# Function to calculate the total potential energy of the system
def total_potential_energy(positions, box_size, epsilon=1.0, sigma=1.0):
    num_particles = positions.shape[0]
    energy = 0.0
    for i in range(num_particles):
        for j in range(i + 1, num_particles):  # Avoid double counting
            # Compute minimum image distance
            delta = positions[i] - positions[j]
            delta -= box_size * np.round(delta / box_size)  # Periodic boundary conditions
            r = np.linalg.norm(delta)
            energy += lennard_jones_potential(r, epsilon, sigma)
    return energy

# Function to perform one step of the Monte Carlo random walk with energy check
def monte_carlo_step(positions, step_size, box_size, epsilon=1.0, sigma=1.0):
    new_positions = positions.copy()
    for i in range(positions.shape[0]):
        # Random step for a single particle
        step = np.random.uniform(-step_size, step_size, positions.shape[1])
        trial_position = apply_periodic_boundary(new_positions[i] + step, box_size)
        
        # Compute energy difference
        original_position = new_positions[i].copy()
        new_positions[i] = trial_position
        energy_new = total_potential_energy(new_positions, box_size, epsilon, sigma)
        new_positions[i] = original_position
        energy_old = total_potential_energy(new_positions, box_size, epsilon, sigma)
        
        delta_energy = energy_new - energy_old
        
        # Accept or reject move
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy):  # Boltzmann criterion
            new_positions[i] = trial_position  # Accept move
            
    return new_positions

# Main function to simulate the random walk
def random_walk_simulation(num_particles=100, box_size=10.0, step_size=1.0, num_steps=100, epsilon=1.0, sigma=1.0):
    # Initialize particle positions
    positions = initialize_particles(num_particles, box_size)
    trajectory = [positions.copy()]
    
    for _ in range(num_steps):
        positions = monte_carlo_step(positions, step_size, box_size, epsilon, sigma)
        trajectory.append(positions.copy())
    
    return trajectory

# Create an animation
def animate_simulation(trajectory, box_size, speed=1.0, save_as_video=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_title("Random Walk Simulation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    points, = ax.plot([], [], 'bo', alpha=0.7)

    def update(frame):
        points.set_data(trajectory[frame][:, 0], trajectory[frame][:, 1])
        ax.set_title(f"Step {frame}")
        return points,

    base_interval = 100  # Base interval in milliseconds
    interval_mod = max(10, int(base_interval / speed))  # Clamp minimum interval to 10 ms
    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=interval_mod, blit=True)
    
    if save_as_video:
        anim.save("random_walk_simulation_v3.gif", writer=PillowWriter(fps=10))  # Save as GIF
    else:
        plt.show()

# Parameters
num_particles = 16     # Number of particles
box_size = 10.0        # Size of the box
step_size = 0.5        # Step size for the random walk
num_steps = 100        # Number of steps to simulate
epsilon = 1.0          # Depth of Lennard-Jones potential well
sigma = 1.0            # Distance at which potential is zero
speed = 5.0            # Animation speed multiplier (1.0 = normal speed)

# Run the simulation
trajectory = random_walk_simulation(num_particles, box_size, step_size, num_steps, epsilon, sigma)

# Generate and display/save the animation
animate_simulation(trajectory, box_size, speed=speed, save_as_video=True)  # Set True to save as GIF
