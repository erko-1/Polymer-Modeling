import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.spatial.distance import cdist


num_monomers = 17
box_size = 25
k_FENE = 40.0
sigma = 1.0
epsilon = 1.0
R_0 = 2.5 * sigma
T = 1.0
k_B = 1.0
num_steps = 10000
step_size = 0.1 
optimal_bond_length = 0.9651375794516794

radius = (optimal_bond_length * num_monomers) / (2 * np.pi)
theta = optimal_bond_length / radius
angles = np.arange(0, num_monomers) * theta

angular_shift = 0
if num_monomers%2==1:
    angular_shift = theta/2

ring1 = np.array([np.cos(angles) * radius, np.sin(angles) * radius, np.zeros(num_monomers)]).T
shift_distance = 2 * radius + optimal_bond_length 
ring2 = np.array([(np.cos(angles-angular_shift) * radius) + shift_distance, np.sin(angles-angular_shift) * radius, np.zeros(num_monomers)]).T


polymer = np.vstack((ring1, ring2))
num_monomers_total = 2 * num_monomers

# Find the closest monomers between the two rings
dist_matrix = cdist(ring1, ring2)  
bridge1, bridge2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
bridge2 += num_monomers  # Adjust index for second ring

ungerade = num_monomers%2

shift1 = num_monomers-1-bridge1
shift2 = bridge2-num_monomers - ungerade

ring12 = np.roll(ring1, shift1, 0)
ring22 = np.roll(ring2, shift2, 0)

polymer = np.vstack((ring12,ring22))

dist_matrix = cdist(ring12, ring22) 
bridge1, bridge2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
bridge2 += num_monomers

def apply_pbc(position, box_size):
    return position - box_size * np.round(position / box_size)

def minimum_image_distance(r1, r2, box_size):
    delta = r1 - r2
    return delta - box_size * np.round(delta / box_size)

def fene_potential(r):
    if r >= R_0:
        return 1e10 
    return -0.5 * k_FENE * R_0**2 * np.log(1 - (r / R_0) ** 2)

def lj_potential(r):
    if r == 0:
        return 1e10
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
    for i in range(num_monomers_total):
        for j in range(num_monomers_total):  
            if i != j:
                r_ij = minimum_image_distance(polymer[i], polymer[j], box_size)
                distance = np.linalg.norm(r_ij)
                distances.append((i, j, distance))
                if (j == (i + 1) or (i == bridge1 and j == bridge2) or (i == bridge2 and j == bridge1)):  
                    energy += tot_potential(distance)
                else:
                    energy += lj_potential(distance)
    return energy 

def monte_carlo_step(polymer):
    i = np.random.randint(num_monomers_total)
    displacement = (np.random.rand(3) - 0.5) * step_size
    new_polymer = polymer.copy()
    new_polymer[i] += displacement
    new_polymer[i] = apply_pbc(new_polymer[i], box_size)

    # Check bond constraints before evaluating energy
    #for j in range(num_monomers_total):
     #   if j == (i + 1) or (i == bridge1 and j == bridge2) or (i == bridge2 and j == bridge1):
      #      r_ij = minimum_image_distance(new_polymer[i], new_polymer[j], box_size)
       #     if np.linalg.norm(r_ij) > R_0:
        #        return polymer  # Reject move immediately

    old_energy = total_energy(polymer)
    new_energy = total_energy(new_polymer)
    dE = new_energy - old_energy

    if dE < 0 or np.random.rand() < np.exp(-dE / (k_B * T)):
        return new_polymer
    return polymer

def compute_center_of_mass(polymer, num_monomers):
    com1 = np.mean(polymer[:num_monomers], axis=0)
    com2 = np.mean(polymer[num_monomers:], axis=0)
    return com1, com2

energy_values = []
angles = []

def update(frame):
    global polymer
    polymer = monte_carlo_step(polymer)
    energy_values.append(total_energy(polymer))

    com1, com2 = compute_center_of_mass(polymer, num_monomers)

    ax.clear()
    ax.set_xlim([-box_size/2, box_size/2])
    ax.set_ylim([-box_size/2, box_size/2])
    ax.set_zlim([-box_size/2, box_size/2])
    ax.set_title(f"Step {frame}")

    # Plot monomers
    ax.scatter(polymer[:bridge1, 0], polymer[:bridge1, 1], polymer[:bridge1, 2], color='r')
    ax.scatter(polymer[(bridge2+1):, 0], polymer[(bridge2+1):, 1], polymer[(bridge2+1):, 2], color='r')
    ax.scatter(polymer[bridge1,0],polymer[bridge1,1], polymer[bridge1,2], color='b')
    ax.scatter(polymer[bridge2,0],polymer[bridge2,1], polymer[bridge2,2], color='b')

    # Plot centers of mass
    ax.scatter(com1[0], com1[1], com1[2], color='black', marker='x', s=10, label="COM1")
    ax.scatter(com2[0], com2[1], com2[2], color='black', marker='x', s=10, label="COM2")

    # Plot bonds within each ring
    for i in range(num_monomers - 1):
        ax.plot([polymer[i, 0], polymer[i + 1, 0]],
                [polymer[i, 1], polymer[i + 1, 1]],
                [polymer[i, 2], polymer[i + 1, 2]], color='b')

    for i in range(num_monomers, num_monomers_total - 1):
        ax.plot([polymer[i, 0], polymer[i + 1, 0]],
                [polymer[i, 1], polymer[i + 1, 1]],
                [polymer[i, 2], polymer[i + 1, 2]], color='b')

    # Close the rings
    ax.plot([polymer[num_monomers - 1, 0], polymer[0, 0]],
            [polymer[num_monomers - 1, 1], polymer[0, 1]],
            [polymer[num_monomers - 1, 2], polymer[0, 2]], color='b')

    ax.plot([polymer[num_monomers_total - 1, 0], polymer[num_monomers, 0]],
            [polymer[num_monomers_total - 1, 1], polymer[num_monomers, 1]],
            [polymer[num_monomers_total - 1, 2], polymer[num_monomers, 2]], color='b')

    # Connect the nearest monomers between the two rings
    ax.plot([polymer[bridge1, 0], polymer[bridge2, 0]],
            [polymer[bridge1, 1], polymer[bridge2, 1]],
            [polymer[bridge1, 2], polymer[bridge2, 2]], 
            color='g', linestyle='dashed')  # Green dashed bridge bond


    midpoint = (polymer[bridge1] + polymer[bridge2]) / 2

    vec1 = com1 - midpoint
    vec2 = com2 - midpoint

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    vec1_n = vec1 / np.linalg.norm(vec1)
    vec2_n = vec2 / np.linalg.norm(vec2)
    
    dot_product = np.dot(vec1_n, vec2_n)
    polar = np.arccos(dot_product)

    angles.append((polar))


    #Plot vectors
    ax.quiver(midpoint[0], midpoint[1], midpoint[2],
              vec1[0], vec1[1], vec1[2], color='purple', length=1, arrow_length_ratio=0.3)

    ax.quiver(midpoint[0], midpoint[1], midpoint[2],
            vec2[0], vec2[1], vec2[2], color='purple', length=1, arrow_length_ratio=0.3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ani = animation.FuncAnimation(fig, update, frames=range(num_steps), interval=50, repeat=False)
plt.show()

plt.figure()
plt.plot(energy_values, label='Total Energy')
plt.xlabel('Monte Carlo Steps')
plt.ylabel('Energy')
plt.title('Total Energy vs. Monte Carlo Steps')
plt.legend()
plt.show()


print("Letzte Abstände:")
for i, j, d in distances[-1000:]:
    if (j - i) == 1:
        print(f'Monomer {i} - Monomer {j}: {d:.3f}')

with open('angles3.txt', 'w') as file:
    for step in range(num_steps):
        file.write(f'{step} {angles[step]}\n')



