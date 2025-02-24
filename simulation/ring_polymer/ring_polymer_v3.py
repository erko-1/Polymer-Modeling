import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Simulation parameters
num_monomers = 10
box_size = 10.0
k_FENE = 40.0
sigma = 1.0
epsilon = 1.0
R_0 = 2.5 * sigma
T = 1.0
k_B = 1.0
num_steps = 2000  # Reduced for visualization
step_size = 1
optimal_bond_length = 0.9651375794516794

# Initialize polymer positions (ring configuration)
radius = (optimal_bond_length * num_monomers) / (2 * np.pi)
theta = optimal_bond_length / radius
angles = np.arange(0, num_monomers) * theta
polymer = np.array([
    np.cos(angles) * radius,
    np.sin(angles) * radius,
    np.zeros(num_monomers)
]).T

def apply_pbc(position, box_size):
    return position - box_size * np.round(position / box_size)

def minimum_image_distance(r1, r2, box_size):
    delta = r1 - r2
    return delta - box_size * np.round(delta / box_size)

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

def total_energy(polymer):
    energy = 0.0
    for i in range(num_monomers):
        j = (i + 1) % num_monomers
        r_ij = minimum_image_distance(polymer[i], polymer[j], box_size)
        distance = np.linalg.norm(r_ij)
        energy += fene_potential(distance)
    for i in range(num_monomers):
        for j in range(i + 2, num_monomers):
            r_ij = minimum_image_distance(polymer[i], polymer[j], box_size)
            distance = np.linalg.norm(r_ij)
            energy += lj_potential(distance)
    return energy

def monte_carlo_step(polymer):
    i = np.random.randint(num_monomers)
    displacement = (np.random.rand(3) - 0.5) * step_size
    new_polymer = polymer.copy()
    new_polymer[i] += displacement
    new_polymer[i] = apply_pbc(new_polymer[i], box_size)
    dE = total_energy(new_polymer) - total_energy(polymer)
    if dE < 0 or np.random.rand() < np.exp(-dE / (k_B * T)):
        return new_polymer
    return polymer

def update(frame):
    global polymer
    polymer = monte_carlo_step(polymer)
    ax.clear()
    ax.set_xlim([-box_size/2, box_size/2])
    ax.set_ylim([-box_size/2, box_size/2])
    ax.set_zlim([-box_size/2, box_size/2])
    ax.set_title(f"Step {frame}")
    ax.plot(polymer[:, 0], polymer[:, 1], polymer[:, 2], marker='o', linestyle='-', color='r')
    ax.plot([polymer[-1, 0], polymer[0, 0]], [polymer[-1, 1], polymer[0, 1]], [polymer[-1, 2], polymer[0, 2]], 'r-')  # Ensure ring closure

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=50)
plt.show()
