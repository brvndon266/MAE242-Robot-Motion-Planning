import numpy as np
import matplotlib.pyplot as plt


# Part d
A = np.array([
    [1,    0.4, 0,    0],
    [-0.6, 1,   0.4,  0],
    [0,    0.4, 1,   -0.6],
    [0,    0,   0.4,  1]
], dtype=float)

B = np.array([[1], [0], [0], [0]], dtype=float)
C = np.array([[0, 0, 0, 1]], dtype=float)

Q = C.T @ C
R = np.array([[1]], dtype=float)

P = Q.copy()

T_vals = []
rho_vals = []
K_vals = []

for T in range(1, 16):
    K = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    Acl = A + B @ K
    eigvals = np.linalg.eigvals(Acl)
    rho = max(abs(eigvals))

    T_vals.append(T)
    rho_vals.append(rho)
    K_vals.append(K)

    print(f"T = {T:2d}, K = {K.flatten()}, rho = {rho:.4f}, stable = {rho < 1}")

    P = Q + K.T @ R @ K + Acl.T @ P @ Acl

plt.figure()
plt.plot(T_vals, rho_vals, marker='o')
plt.axhline(1, linestyle='--', color='red')
plt.xlabel("Horizon T")
plt.ylabel("Spectral radius of A + B K_T")
plt.title("Closed-loop spectral radius vs horizon T")
plt.legend(["Spectral radius", "Stability threshold"])
plt.grid(True)
plt.show()

# Part f)

Q_base = C.T @ C
R = np.array([[1]], dtype=float)

rho_list = np.arange(1.0, 0.0, -0.1)
Ts_list = []

for rho in rho_list:
    Q = rho * Q_base
    P = Q.copy()

    Ts = None

    for T in range(1, 16):
        K = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        Acl = A + B @ K
        spectral_radius = max(abs(np.linalg.eigvals(Acl)))

        if spectral_radius < 1 and Ts is None:
            Ts = T

        P = Q + K.T @ R @ K + Acl.T @ P @ Acl

    Ts_list.append(Ts)

    print(f"rho = {rho:.1f}, Ts = {Ts}")

plt.figure()
plt.plot(rho_list, Ts_list, marker='o')
plt.xlabel(r"$\rho$")
plt.ylabel(r"$T_s(\rho)$")
plt.title(r"Minimum stabilizing horizon $T_s(\rho)$ vs $\rho$")
plt.grid(True)
plt.show()