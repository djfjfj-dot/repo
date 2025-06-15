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

# Simulated Annealing aided by additive TMCMC for LJ potential
def optimize_lj_tmcmc(n_particles=5, N=10000, epsilon=1.0, sigma=1.0):
    # Initialize random 3D coordinates
    theta = np.random.uniform(-1.0, 1.0, (n_particles, 3))
    best_theta = theta.copy()
    best_energy = lj_potential(theta, epsilon, sigma)

    # Step size constants
    a1 = 0.1
    a2 = 0.1

    history = []

    for t in range(1, N + 1):
        # Perturbation variables
        e = np.random.normal(0, 1, size=(n_particles, 3))
        b = np.random.choice([-1, 1], size=(n_particles, 3))

        # Additive TMCMC proposal
        theta_tilde = theta + b * a1 * np.abs(e)

        # Energy evaluation
        current_energy = lj_potential(theta, epsilon, sigma)
        proposed_energy = lj_potential(theta_tilde, epsilon, sigma)

        # Log-likelihood difference (using negative potential)
        ell_diff = -proposed_energy + current_energy

        # Cooling schedule
        tau_t = 1 / np.log(np.log(t + 2)) if t >= 3 else 1 / t

        # Acceptance probability
        alpha = min(1, np.exp(ell_diff / tau_t))

        # Accept or reject
        if np.random.rand() < alpha:
            theta = theta_tilde
            if proposed_energy < best_energy:
                best_theta = theta_tilde
                best_energy = proposed_energy

        history.append(best_energy)

    return best_theta, best_energy, history

# Run the optimizer
best_positions, min_energy, energy_history = optimize_lj_tmcmc(n_particles=5, N=5000)

# Output result
print("Best particle positions:\n", best_positions)
print("Minimum Lennard-Jones potential:", min_energy)
