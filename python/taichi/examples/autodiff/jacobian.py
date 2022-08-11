import matplotlib.pyplot as plt
import numpy as np

import taichi as ti

ti.init()

N_param = 50
N_loss = 100
params = ti.field(float, shape=N_param, needs_dual=True, needs_grad=True)
losses = ti.field(float, shape=N_loss, needs_dual=True, needs_grad=True)


@ti.kernel
def func():
    for j in range(N_loss):
        for i in range(N_param):
            losses[j] = ti.sin(j / 16 * np.pi) * ti.sin(
                i / 16 * np.pi) * params[i]


# Compute Jacobian matrix via forward mode autodiff
jacobian_fwd = np.zeros((N_loss, N_param))
seed = [0.0 for _ in range(N_param)]
for n_p in range(N_param):
    # Compute Jacobian-vector product (Jvp) N_param times
    seed[n_p] = 1.0
    with ti.ad.FwdMode(loss=losses, param=params, seed=seed):
        func()
    jacobian_fwd[:, n_p] = losses.dual.to_numpy()
    seed[n_p] = 0.0

# Compute Jacobian matrix via reverse mode autodiff
jacobian_rev = np.zeros((N_loss, N_param))
for n_l in range(N_loss):
    # Compute vector-Jacobian product (vJp) N_loss times
    losses.grad[n_l] = 1.0
    func.grad()
    jacobian_rev[n_l, :] = params.grad.to_numpy()
    params.grad.fill(0)
    losses.grad[n_l] = 0.0

plt.subplot(121)
plt.imshow(jacobian_fwd)
plt.title("Jacobian Matrix \n (via forward mode)")
plt.subplot(122)
plt.imshow(jacobian_rev)
plt.title("Jacobian Matrix \n (via reverse mode)")
plt.show()
