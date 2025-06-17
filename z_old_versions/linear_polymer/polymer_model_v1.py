import numpy as np

def gen_chain(N, R0):
    x = np.linspace(1, (N-1)*0.8*R0, num=N)
    y = np.zeros(N)
    z = np.zeros(N)
    return np.column_stack((x, y, z))

def lj(rij2):
    sig_by_r6 = np.power(sigma/rij2, 3)
    sig_by_r12 = np.power(sig_by_r6, 2)
    lje = 4.0 * epsilon * (sig_by_r12 - sig_by_r6)
    return lje

def fene(rij2):
    return (-0.5 * K * R**2 * np.log(1-((np.sqrt(rij2) - r0) / R)**2))

def total_energy(coord):
    # Non-bonded
    e_nb = 0
    for i in range(N):
        for j in range(i-1):
            ri = coord[i]
            rj = coord[j]
            rij = ri - rj
            rij2 = np.dot(rij, rij)
            if (np.sqrt(rij2) < rcutoff):
                e_nb += lj(rij2)
    # Bonded
    e_bond = 0
    for i in range(1, N):
        ri = coord[i]
        rj = coord[i-1]
        rij = ri - rj
        rij2 = np.dot(rij, rij)
        e_bond += fene(rij2)
    return e_nb + e_bond

def move(coord):
    trial = np.ndarray.copy(coord)
    for i in range(N):
        delta = (2.0 * np.random.rand(3) - 1) * max_delta
        trial[i] += delta
    return trial

def accept(delta_e):
    beta = 1.0/T
    if delta_e <= 0.0:
        return True
    random_number = np.random.rand(1)
    p_acc = np.exp(-beta*delta_e)
    if random_number < p_acc:
        return True
    return False


if __name__ == "__main__":

    # FENE parameters
    K = 40
    R = 0.3
    r0 = 0.7

    # LJ parameters
    sigma = r0/0.33
    epsilon = 1.0

    # MC parameters
    N = 50 # number of particles
    rcutoff = 2.5*sigma
    max_delta = 0.01
    n_steps = 10000000
    T = 0.5

    coord = gen_chain(N, R)
    energy_current = total_energy(coord)

    traj = open('traj.xyz', 'w') 

    for step in range(n_steps):
        if step % 1000 == 0:
            traj.write(str(N) + '\n\n')
            for i in range(N):
                traj.write("C %10.5f %10.5f %10.5f\n" % (coord[i][0], coord[i][1], coord[i][2]))
            print(step, energy_current)
        coord_trial = move(coord)
        energy_trial = total_energy(coord_trial)
        delta_e =  energy_trial - energy_current
        if accept(delta_e):
            coord = coord_trial
            energy_current = energy_trial

    traj.close()