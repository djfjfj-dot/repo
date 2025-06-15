import numpy as np

# Lennard-Jones potential function
def lj_potential(positions, epsilon=1.0, sigma=1.0):
    n_particles = positions.shape[0]
    potential = 0.0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r = np.linalg.norm(positions[i] - positions[j])
            if r != 0:
                sr6 = (sigma / r) ** 6
                sr12 = sr6 ** 2
                potential += 4 * epsilon * (sr12 - sr6)
    return potential

# Enhanced mixing TMCMC Simulated Annealing
def optimize_lj_tmcmc_enhanced(n_particles=5, N=10000, epsilon=1.0, sigma=1.0):
    theta = np.random.uniform(-1.0, 1.0, (n_particles, 3))
    best_theta = theta.copy()
    best_energy = lj_potential(theta)

    a1 = 0.1
    a2 = 0.1
    history = []

    for t in range(1, N + 1):
        # ------- Stage 1: Basic TMCMC Move -------
        e1 = np.random.normal(0, 1, (n_particles, 3))
        e2 = np.random.uniform(1e-5, 1.0)  # To avoid log(0)
        b = np.random.choice([-1, 1], size=(n_particles, 3))
        U1 = np.random.uniform()

        if U1 < 0.5:
            theta_tilde = theta + b * a1 * np.abs(e1)
            J = 1  # Additive Jacobian
        else:
            theta_tilde = theta * np.exp(b * a1 * np.abs(e1))
            J = np.abs(e1).sum()  # Multiplicative Jacobian

        E_current = lj_potential(theta)
        E_tilde = lj_potential(theta_tilde)

        ell_diff1 = -E_tilde + E_current
        tau_t = 1 / np.log(np.log(t + 2)) if t >= 100 else 1 / t
        alpha1 = min(1, np.exp((ell_diff1 + np.log(J)) / tau_t))

        if np.random.rand() < alpha1:
            theta = theta_tilde
            if E_tilde < best_energy:
                best_energy = E_tilde
                best_theta = theta

        # ------- Stage 2: Further Mixing -------
        e3 = np.random.normal(0, 1, (n_particles, 3))
        U2 = np.random.uniform()

        if U2 < 0.5:
            u1 = np.random.uniform()
            if u1 < 0.5:
                theta_star = theta + a1 * np.abs(e3)
            else:
                theta_star = theta - a1 * np.abs(e3)
            J = 1
        else:
            e4 = np.random.choice([-1, 1], size=(n_particles, 3))
            u2 = np.random.uniform()
            if u2 < 0.5:
                theta_star = theta * e4
                J = np.abs(e4).sum()
            else:
                theta_star = theta / e4
                J = (1 / np.abs(e4)).sum()

        E_star = lj_potential(theta_star)
        ell_diff2 = -E_star + lj_potential(theta)
        tau_t2 = 1 / np.log(np.log(t + 2)) if t >= 100 else 1 / t
        alpha2 = min(1, np.exp((ell_diff2 + np.log(J)) / tau_t2))

        if np.random.rand() < alpha2:
            theta = theta_star
            if E_star < best_energy:
                best_energy = E_star
                best_theta = theta

        history.append(best_energy)

    return best_theta, best_energy, history

# Run the enhanced optimizer
best_positions, min_energy, energy_history = optimize_lj_tmcmc_enhanced(n_particles=5, N=5000)

# Print results
print("Best particle positions:\n", best_positions)
print("Minimum Lennard-Jones potential:", min_energy)
