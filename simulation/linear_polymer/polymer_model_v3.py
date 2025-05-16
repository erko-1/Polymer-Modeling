import numpy as np
import matplotlib.pyplot as plt

def gen_chain(N, R0):
    x = np.linspace(1, (N-1)*0.8*R0, num=N)
    y = np.zeros(N)
    z = np.zeros(N)
    return np.column_stack((x, y, z))

def lj(rij2):
    if rij2 == 0:
        return 0
    sig_by_r6 = np.power(sigma / np.sqrt(rij2), 6)
    sig_by_r12 = sig_by_r6 ** 2
    return 4.0 * epsilon * (sig_by_r12 - sig_by_r6)

def fene(rij2):
    r = np.sqrt(rij2)
    if r > r0 + R:
        return np.inf  # Prevent unphysical stretching
    return -0.5 * K * R**2 * np.log(1 - ((r - r0) / R)**2)

def total_energy(coord):
    e_nb = 0.0
    for i in range(N):
        for j in range(i):  # Fixed indexing issue
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

def accept(delta_e):
    beta = 1.0 / T
    if delta_e <= 0.0:
        return True
    return np.random.rand() < np.exp(-beta * delta_e)

if __name__ == "__main__":
    # FENE parameters
    K = 40
    R = 0.3
    r0 = 0.7

    # LJ parameters
    sigma = r0 / 0.33
    epsilon = 1.0

    # MC parameters
    N = 50
    rcutoff = 2.5 * sigma
    max_delta = 0.01
    n_steps = 1000000
    T = 0.5

    coord = gen_chain(N, R)
    energy_current = total_energy(coord)
    energy_history = []

    with open('traj.xyz', 'w') as traj:
        for step in range(n_steps):
            if step % 1000 == 0:
                traj.write(f"{N}\n\n")
                for atom in coord:
                    traj.write(f"C {atom[0]:10.5f} {atom[1]:10.5f} {atom[2]:10.5f}\n")
                print(step, energy_current)
            
            energy_history.append(energy_current)
            coord_trial = move(coord)
            energy_trial = total_energy(coord_trial)
            delta_e = energy_trial - energy_current
            
            if accept(delta_e):
                coord = coord_trial
                energy_current = energy_trial
    
    # Plot energy history
    plt.plot(energy_history)
    plt.xlabel('Step')
    plt.ylabel('Total Energy')
    plt.title('Total Energy Over Time')
    plt.show()
