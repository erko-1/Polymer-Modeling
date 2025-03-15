import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

num_monomers = 10
box_size = 10.0
k_FENE = 40.0
sigma = 1.0
epsilon = 1.0
R_0 = 2.5 * sigma
T = 1
k_B = 1.0
num_steps = 1000
step_size = 0.05 
optimal_bond_length = 0.9651375794516794


# Ring
radius = (optimal_bond_length * num_monomers) / (2 * np.pi)
theta = optimal_bond_length / radius
angles = np.arange(0, num_monomers) * theta
polymer = np.array([np.cos(angles) * radius, np.sin(angles) * radius, np.zeros(num_monomers)]).T

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

def tot_potential(r):
    return fene_potential(r) + lj_potential(r)

distances = []

def total_energy(polymer):
    energy = 0.0
    global distances
    distances.clear()
    
    for i in range(num_monomers):
        for j in range(num_monomers):  # j startet wieder bei 0
            if i != j:  # Damit keine Selbstwechselwirkung passiert
                r_ij = minimum_image_distance(polymer[i], polymer[j], box_size)
                distance = np.linalg.norm(r_ij)
                distances.append((i, j, distance))

                if j == (i + 1):
                    energy += tot_potential(distance)
                else:
                    energy += lj_potential(distance)

    return energy

def monte_carlo_step(polymer):
    i = np.random.randint(num_monomers)
    displacement = (np.random.rand(3) - 0.5) * step_size
    new_polymer = polymer.copy()
    new_polymer[i] += displacement
    new_polymer[i] = apply_pbc(new_polymer[i], box_size)
    
    old_energy = total_energy(polymer)
    new_energy = total_energy(new_polymer)
    dE = new_energy - old_energy
    
    if dE < 0 or np.random.rand() < np.exp(dE / (-1 * k_B * T)):
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
    ax.plot(polymer[:, 0], polymer[:, 1], polymer[:, 2], marker='o', linestyle='-', color='r')
    ax.plot([polymer[-1, 0], polymer[0, 0]], [polymer[-1, 1], polymer[0, 1]], [polymer[-1, 2], polymer[0, 2]], 'r-')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ani = animation.FuncAnimation(fig, update, frames=range(num_steps), interval=50, repeat=False)
plt.show()

# Ausgabe der letzten 10 Abstände nach der Simulation
print("Letzte 10 Abstände:")
for i, j, d in distances[-10:]:  
    print(f'Monomer {i} - Monomer {j}: {d:.3f}')

# Plot total energy
plt.figure()
plt.plot(energy_values, label='Total Energy')
plt.xlabel('Monte Carlo Steps')
plt.ylabel('Energy')
plt.title('Total Energy vs. Monte Carlo Steps')
plt.legend()
plt.show()