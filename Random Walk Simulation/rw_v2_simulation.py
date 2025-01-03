import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class RandomWalkSimulation:
    def __init__(self, num_particles, box_size, step_size, min_distance, max_steps):
        self.num_particles = num_particles
        self.box_size = box_size  # Size of the simulation box
        self.step_size = step_size  # Maximum step size for a particle
        self.min_distance = min_distance  # Minimum distance between particles
        self.max_steps = max_steps  # Number of simulation steps
        self.positions = self.initialize_positions()  # Initialize particle positions

    def initialize_positions(self):
        # Randomly initialize particle positions ensuring no overlap
        positions = []
        while len(positions) < self.num_particles:
            new_pos = np.random.uniform(0, self.box_size, size=2)
            if all(np.linalg.norm(new_pos - pos) > self.min_distance for pos in positions):
                positions.append(new_pos)
        return np.array(positions)

    def periodic_boundary_conditions(self, position):
        # Apply periodic boundary conditions
        return position % self.box_size

    def is_valid_move(self, new_pos, current_idx):
        # Check if the new position is valid (no overlap with other particles)
        for i, pos in enumerate(self.positions):
            if i != current_idx and np.linalg.norm(new_pos - pos) < self.min_distance:
                return False
        return True

    def update(self, frame):
        # Perform one step of the simulation (Monte Carlo move)
        for i in range(self.num_particles):
            # Propose a random move
            move = np.random.uniform(-self.step_size, self.step_size, size=2)
            new_pos = self.positions[i] + move
            new_pos = self.periodic_boundary_conditions(new_pos)  # Apply boundary conditions

            # Accept the move if valid
            if self.is_valid_move(new_pos, i):
                self.positions[i] = new_pos

        # Clear previous plot and plot new particle positions
        self.ax.clear()
        self.ax.set_xlim(0, self.box_size)
        self.ax.set_ylim(0, self.box_size)
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.set_title("Random Walk of Particles")
        self.ax.scatter(self.positions[:, 0], self.positions[:, 1], c='blue', s=50, label='Particles')
        return self.ax,

    def run_simulation(self, gif_filename):
        # Create the plot
        fig, ax = plt.subplots(figsize=(6, 6))
        self.ax = ax

        # Create the animation
        ani = FuncAnimation(fig, self.update, frames=self.max_steps, interval=100, repeat=False)

        # Save the animation as a GIF
        writer = PillowWriter(fps=30)
        ani.save(gif_filename, writer=writer)

# Parameters
num_particles = 10  # Number of particles
box_size = 10.0  # Size of the simulation box
step_size = 0.1  # Maximum step size for particles
min_distance = 0.1  # Minimum distance between particles
max_steps = 1000  # Number of simulation steps
gif_filename = "random_walk_simulation_v2.gif"  # Output GIF filename

# Run the simulation and save as a GIF
simulation = RandomWalkSimulation(num_particles, box_size, step_size, min_distance, max_steps)
simulation.run_simulation(gif_filename)

print(f"GIF saved as {gif_filename}")
