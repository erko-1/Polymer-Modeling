import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def gen_chain(N, R0):
    x = np.linspace(1, (N-1)*0.8*R0, num=N)
    y = np.zeros(N)
    z = np.zeros(N)
    return np.column_stack((x, y, z))

def lj(rij2, sigma=1, epsilon=1):
    if rij2 == 0:
        return np.inf
    if rij2 < 2**(1/6) * sigma:
        sr6 = (sigma / rij2)**6
        sr12 = sr6**2
        return 4 * epsilon * (sr12 - sr6) + epsilon
    return 0


def fene(rij2, R=2.5, K=40, r0 = 0):
    if rij2 > r0 + R:
        return np.inf  # Prevent unphysical stretching
    return -0.5 * K * R**2 * np.log(1 - ((r - r0) / R)**2)

def total_energy(coord, rcutoff = 2**(1/6)):
    e_nb = 0.0
    for i in range(N):
        for j in range(i): 
            rij = coord[i] - coord[j]
            rij2 = np.dot(rij, rij)
            if rij2 < rcutoff**2:
                e_nb += lj(rij2)
    
    e_bond = 0.0
    for i in range(1, N):
        rij = coord[i] - coord[i-1]
        rij2 = np.dot(rij, rij)
        e_bond += fene(rij2)
    
    return e_nb + e_bond

def move(coord):
    trial = np.copy(coord)
    for i in range(N):
        delta = (2.0 * np.random.rand(3) - 1) * max_delta
        trial[i] += delta
    return trial

def accept(delta_e, k_B=1, T=1):
    beta = 1.0 / (k_B * T)
    if delta_e <= 0.0:
        return True
    return np.random.rand() < np.exp(-beta * delta_e)

def update_visualization(frame):
    global coord, energy_current, past_coords
    for _ in range(500):  # Increase number of MC steps before updating visualization
        coord_trial = move(coord)
        energy_trial = total_energy(coord_trial)
        delta_e = energy_trial - energy_current
        if accept(delta_e):
            coord = coord_trial
            energy_current = energy_trial
    past_coords.append(np.copy(coord))  # Store past coordinates
    
    ax.clear()
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    
    for past in past_coords[-10:]:  # Keep last 10 states for trajectory effect
        ax.plot(past[:, 0], past[:, 1], past[:, 2], 'o-', alpha=0.3, markersize=3)
    
    ax.plot(coord[:, 0], coord[:, 1], coord[:, 2], 'o-', markersize=5, color='red')

if __name__ == "__main__":
    # Pot. parameters
    T = 1
    k_B = 1
    sigma = 1

    K = 40 * k_B * T
    R = 2.5*sigma
    r0 = 0


    # MC parameters
    N = 10
    rcutoff = 2**(1/6) * sigma
    max_delta = 0.01
    n_steps = 100

    coord = gen_chain(N, R)
    energy_current = total_energy(coord)
    past_coords = []  # Store past positions

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ani = animation.FuncAnimation(fig, update_visualization, frames=100, interval=50)  # Faster updates
    plt.show()
