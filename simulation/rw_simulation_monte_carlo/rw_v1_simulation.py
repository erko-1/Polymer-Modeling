import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Function to initialize particle positions
def initialize_particles(num_particles, box_size):
    return np.random.uniform(0, box_size, (num_particles, 2))  # 2D positions

# Function to apply periodic boundary conditions
def apply_periodic_boundary(position, box_size):
    return position % box_size

# Function to perform one step of the Monte Carlo random walk
def monte_carlo_step(positions, step_size, box_size):
    # Random step in each direction for each particle
    steps = np.random.uniform(-step_size, step_size, positions.shape)
    new_positions = positions + steps
    # Apply periodic boundary conditions
    new_positions = apply_periodic_boundary(new_positions, box_size)
    return new_positions

# Main function to simulate the random walk
def random_walk_simulation(num_particles=100, box_size=10.0, step_size=1.0, num_steps=100):
    # Initialize particle positions
    positions = initialize_particles(num_particles, box_size)
    trajectory = [positions.copy()]
    
    for _ in range(num_steps):
        positions = monte_carlo_step(positions, step_size, box_size)
        trajectory.append(positions.copy())
    
    return trajectory

# Create an animation
def animate_simulation(trajectory, box_size, speed=1.0, save_as_video=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_title("Random Walk of Particles")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    points, = ax.plot([], [], 'bo', alpha=0.7)

    def update(frame):
        points.set_data(trajectory[frame][:, 0], trajectory[frame][:, 1])
        ax.set_title(f"Step {frame}")
        return points,

    # Adjust the speed of the animation by modifying the interval
    interval = int(100 / speed)  # Base interval is 100 ms; modify by speed factor
    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=True)
    
    if save_as_video:
        anim.save("random_walk_simulation.gif", writer=PillowWriter(fps=10))  # Save as GIF
    else:
        plt.show()

# Parameters
num_particles = 16     # Number of particles
box_size = 10.0        # Size of the box
step_size = 0.1        # Step size for the random walk
num_steps = 100        # Number of steps to simulate
speed = 1            # Animation speed multiplier (1.0 = normal speed)

# Run the simulation
trajectory = random_walk_simulation(num_particles, box_size, step_size, num_steps)

# Generate and display/save the animation
animate_simulation(trajectory, box_size, speed=speed, save_as_video=False)  # Set True to save as GIF
