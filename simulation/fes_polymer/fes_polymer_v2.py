import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

num_monomers = 10  
box_size = 15.0
k_FENE = 40.0
sigma = 1.0
epsilon = 1.0
R_0 = 2.5 * sigma
T = 1
k_B = 1.0
num_steps = 1
step_size = 0.5 
optimal_bond_length = 0.9651375794516794

radius = (optimal_bond_length * num_monomers) / (2 * np.pi)
theta = optimal_bond_length / radius
angles = np.arange(0, num_monomers) * theta

ring1 = np.array([np.cos(angles) * radius, np.sin(angles) * radius, np.zeros(num_monomers)]).T

shift_distance = 2 * radius + optimal_bond_length 
ring2 = np.array([(np.cos(angles) * radius) + shift_distance, np.sin(angles) * radius, np.zeros(num_monomers)]).T

polymer = np.vstack((ring1, ring2))
num_monomers_total = 2*num_monomers

# Connect the two rings at their closest points (bridge monomers)
bridge1, bridge2 = num_monomers - 1, num_monomers  # Last monomer of first ring and first of second

def apply_pbc(position, box_size):
    return position - box_size * np.round(position / box_size)

def minimum_image_distance(r1, r2, box_size):
    delta = r1 - r2
    return delta - box_size * np.round(delta / box_size)

def fene_potential(r):
    if r >= R_0 * 0.999: 
        return np.inf
    return -0.5 * k_FENE * R_0**2 * np.log(1 - (r / R_0) ** 2 + 1e-8)


def lj_potential(r):
    if r < 1e-5:
        return np.inf
    if r < 2**(1/6) * sigma:
        sr6 = (sigma / r)**6
        sr12 = sr6**2
        return 4 * epsilon * (sr12 - sr6) + epsilon
    return 0

def tot_potential(r):
    return fene_potential(r) + lj_potential(r)

def total_energy(polymer):
    energy = 0.0
    for i in range(num_monomers_total):
        for j in range(num_monomers_total):
            if i != j:
                r_ij = minimum_image_distance(polymer[i], polymer[j], box_size)
                distance = np.linalg.norm(r_ij)
                if j == (i + 1) or (i == bridge1 and j == bridge2): 
                    energy += tot_potential(distance)
                else:
                    energy += lj_potential(distance)
            else:
                continue
        return energy

def monte_carlo_step(polymer):
    i = np.random.randint(num_monomers_total)
    displacement = (np.random.rand(3) - 0.5) * step_size
    new_polymer = polymer.copy()
    new_polymer[i] += displacement
    new_polymer[i] = apply_pbc(new_polymer[i], box_size)
    
    old_energy = total_energy(polymer)
    new_energy = total_energy(new_polymer)
    dE = new_energy - old_energy
    
    if dE < 0 or np.random.rand() < np.exp(-dE/(k_B * T)):
        return new_polymer
    return polymer



energy_values = []

def update(frame):
    global polymer
    polymer = monte_carlo_step(polymer)
    energy_values.append(total_energy(polymer))

    ax.clear()
    ax.set_xlim([-box_size/2, box_size/2])
    ax.set_ylim([-box_size/2, box_size/2])
    ax.set_zlim([-box_size/2, box_size/2])
    ax.set_title(f"Step {frame}")

    # Plot monomers
    ax.scatter(polymer[:, 0], polymer[:, 1], polymer[:, 2], color='r')

    # Plot bonds within each ring
    for i in range(num_monomers - 1):
        ax.plot([polymer[i, 0], polymer[i + 1, 0]],
                [polymer[i, 1], polymer[i + 1, 1]],
                [polymer[i, 2], polymer[i + 1, 2]], color='b')

    for i in range(num_monomers, num_monomers_total - 1):
        ax.plot([polymer[i, 0], polymer[i + 1, 0]],
                [polymer[i, 1], polymer[i + 1, 1]],
                [polymer[i, 2], polymer[i + 1, 2]], color='b')

    # Connect the last monomer of ring1 to the first monomer of ring2 (bridge bond)
    ax.plot([polymer[bridge1, 0], polymer[bridge2, 0]],
            [polymer[bridge1, 1], polymer[bridge2, 1]],
            [polymer[bridge1, 2], polymer[bridge2, 2]], color='g', linestyle='dashed')  # Green dashed bridge
    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ani = animation.FuncAnimation(fig, update, frames=range(num_steps), interval=50, repeat=False)
plt.show()

# Plot total energy
plt.figure()
plt.plot(energy_values, label='Total Energy')
plt.xlabel('Monte Carlo Steps')
plt.ylabel('Energy')
plt.title('Total Energy vs. Monte Carlo Steps')
plt.legend()
plt.show()
