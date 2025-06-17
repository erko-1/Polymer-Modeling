import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


num_monomers = 17
box_size = 10.0
k_FENE = 40.0
sigma = 1.0
epsilon = 1.0
R_0 = 2.5 * sigma
T = 1
k_B = 1.0
num_steps = 100000
step_size = 0.05
optimal_bond_length = 0.9651375794516794

radius = (optimal_bond_length * num_monomers) / (2 * np.pi)
theta = optimal_bond_length / radius
angles = np.arange(0, num_monomers) * theta
polymer = np.array([np.cos(angles) * radius, np.sin(angles) * radius, np.zeros(num_monomers)]).T

bonded_pairs = [(i, (i + 1) % num_monomers) for i in range(num_monomers)]

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
    if r < 2**(1 / 6) * sigma:
        sr6 = (sigma / r) ** 6
        sr12 = sr6 ** 2
        return 4 * epsilon * (sr12 - sr6) + epsilon
    return 0

def tot_potential(r):
    return fene_potential(r) + lj_potential(r)

def total_energy(polymer):
    energy = 0.0
    for i in range(num_monomers):
        for j in range(i+1, num_monomers):
            if j != i:
                r_ij = minimum_image_distance(polymer[i], polymer[j], box_size)
                distance = np.linalg.norm(r_ij)
                if (i, j) in bonded_pairs or (j, i) in bonded_pairs:
                    energy += tot_potential(distance)
                if (i, j) not in bonded_pairs or (j, i) not in bonded_pairs:
                    energy += lj_potential(distance)
    return energy

def monte_carlo_step(polymer):
    i = np.random.randint(num_monomers)
    old_energy = total_energy(polymer)

    displacement = (np.random.rand(3) - 0.5) * step_size
    new_polymer = polymer.copy()
    new_polymer[i] += displacement
    new_polymer[i] = apply_pbc(new_polymer[i], box_size)

    new_energy = total_energy(new_polymer)
    dE = new_energy - old_energy

    if dE < 0 or np.random.rand() < np.exp(-dE / (k_B * T)):
        return new_polymer
    return polymer

def compute_gyration_tensor_ring(polymer):
    com = np.mean(polymer, axis=0)
    rel_positions = polymer - com
    gyration_tensor = np.zeros((3, 3))
    for pos in rel_positions:
        for i in range(3):
            for j in range(3):
                gyration_tensor[i, j] += pos[i] * pos[j]
    gyration_tensor /= len(polymer)
    return gyration_tensor


energy_values = []
autocorrelations = []
Rg_squared_ring = []
eig_values_single_ring = []

for frame in range(num_steps):
    polymer = monte_carlo_step(polymer)
    energy_values.append(total_energy(polymer))

    if frame != 0 and frame % 100 == 0:
        gyr = compute_gyration_tensor_ring(polymer)
        Rg_squared_ring.append(np.sum(np.linalg.eigvalsh(gyr)))
        eig_values_single_ring.append(np.linalg.eigvalsh(gyr))
    print(f"Step {frame}")

with open('SR_Rg_squared6 .txt', 'w') as file:
    file.write("Step\tRg_squared\n")
    for i in range(len(Rg_squared_ring)):
        start = i * 100
        end = (i + 1) * 100
        file.write(f'{start}-{end}\t{Rg_squared_ring[i]}\n')

with open('SR_Eigenvalues6.txt', 'w') as f:
    f.write("Step\tEig_val_1\tEig_val_2\tEig_val_3\n")
    for i in range(len(eig_values_single_ring)):
        eig_vals_str = '\t'.join(map(str, eig_values_single_ring[i]))
        f.write(f'{i * 100}-{(i + 1) * 100}\t{eig_vals_str}\n')

plt.figure()
plt.plot(energy_values, label='Total Energy')
plt.xlabel('Monte Carlo Steps')
plt.ylabel('Energy')
plt.title('Total Energy vs. Monte Carlo Steps')
plt.legend()
plt.show()
