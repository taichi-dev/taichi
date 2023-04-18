""" phase field model for snow dendrite evolution, which simulates the snow growing from water vapor.
    space discretization: finite difference method, time integration: Runge-Kutta method
    repo's link: https://github.com/mo-hanxuan/Snow-PhaseField
    more details about physical interpretation refer to [Physica D 63(3-4): 410-423]"""
import numpy as np

import taichi as ti


@ti.data_oriented
class Dendrite:
    def __init__(
        self,
        dx=0.03,  # grid space
        dt=2.0e-4,  # time step
        n=512,  # field shape
        dtype=ti.f64,  # data type
        n_fold_symmetry=6,  # dendrites number
        angle0=0.0,  # initial angle
    ):
        self.n = n  # the size of the field
        ### phase field indicate solid phase (phi=1) and water vapor phase (phi=0)
        self.phi = ti.field(dtype=dtype, shape=(n, n))
        self.temperature = ti.field(dtype=dtype, shape=(n, n))
        self.phi_old = ti.field(dtype=dtype, shape=(n, n))
        self.temperature_old = ti.field(dtype=dtype, shape=(n, n))
        self.dEnergy_dGrad_term1 = ti.Vector.field(2, dtype, shape=(n, n))
        self.epsilons = ti.field(dtype=dtype, shape=(n, n))  # anisotropic gradient energy coefficient
        self.phiRate = ti.Vector.field(4, dtype=dtype, shape=(n, n))  # rate of phi, with RK4 method
        self.temperatureRate = ti.Vector.field(4, dtype=dtype, shape=(n, n))

        self.dx = dx
        self.dt = dt  # maximum dt before going unstable is 2.9e-4
        self.mobility = 128.0e2

        self.grad_energy_coef = 0.006  # magnitude of gradient energy coefficient
        self.latent_heat_coef = 1.5  # latent heat coefficient
        self.aniso_magnitude = 0.12  # the magnitude of anisotropy
        self.n_fold_symmetry = n_fold_symmetry  # n-fold rotational symmetry of the crystalline structure
        self.angle0 = angle0 / 180.0 * np.pi  # initial tilt angle of the crystal

        ### m(T) = (α / π) * arctan[γ(Te - T)], m determines the relative potential between water vapor and crystal
        self.alpha = 0.9 / np.pi  # α
        self.gamma = 10.0  # γ
        self.temperature_equi = 1.0  # temperature of equilibrium state

        ### parameters for RK4 method
        self.dtRatio_rk4 = ti.field(ti.f64, shape=(4))
        self.dtRatio_rk4.from_numpy(np.array([0.0, 0.5, 0.5, 1.0]))
        self.weights_rk4 = ti.field(ti.f64, shape=(4))
        self.weights_rk4.from_numpy(np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]))

        self.showFrameFrequency = int(4 * 1.0e-4 / self.dt)

    @ti.kernel
    def initialize(
        self,
    ):
        radius = 4.0  # 1.
        center = ti.Vector([self.n // 2, self.n // 2])
        for I in ti.grouped(self.phi):
            if ((ti.Vector([I[0], I[1]]) - center) ** 2).sum() < radius**2:
                self.phi[I] = 1.0
            else:
                self.phi[I] = 0.0
            self.phi_old[I] = self.phi[I]

            self.temperature[I] = 0.0  # temperature
            self.temperature_old[I] = self.temperature[I]

    @ti.func
    def neighbor_index(self, i, j):
        """use periodic boundary condition to get neighbor index"""
        im = i - 1 if i - 1 >= 0 else self.n - 1
        jm = j - 1 if j - 1 >= 0 else self.n - 1
        ip = i + 1 if i + 1 < self.n else 0
        jp = j + 1 if j + 1 < self.n else 0
        return im, jm, ip, jp

    @ti.kernel
    def rk4_intermediate_update(self, rk_loop: int):
        """update field variables at the intermediate step of RK4"""
        phi, phi_old, temperature, temperature_old, dt = ti.static(
            self.phi, self.phi_old, self.temperature, self.temperature_old, self.dt
        )
        for I in ti.grouped(phi):
            phi[I] = phi_old[I] + self.dtRatio_rk4[rk_loop] * dt * self.phiRate[I][rk_loop - 1]
            temperature[I] = temperature_old[I] + self.dtRatio_rk4[rk_loop] * dt * self.temperatureRate[I][rk_loop - 1]

    @ti.kernel
    def get_rate(self, rk_loop: int):
        """get rate of phi and temperature"""
        (
            phi,
            temperature,
            dx,
            mobility,
            aniso_magnitude,
            epsilons,
            n_fold_symmetry,
            grad_energy_coef,
            angle0,
        ) = ti.static(
            self.phi,
            self.temperature,
            self.dx,
            self.mobility,
            self.aniso_magnitude,
            self.epsilons,
            self.n_fold_symmetry,
            self.grad_energy_coef,
            self.angle0,
        )
        ### first, get epsilons and dEnergy_dGrad_term1
        for i, j in phi:
            im, jm, ip, jp = self.neighbor_index(i, j)
            grad = ti.Vector(
                [
                    (phi[ip, j] - phi[im, j]) / (2.0 * dx),
                    (phi[i, jp] - phi[i, jm]) / (2.0 * dx),
                ]
            )
            gradNorm = (grad**2).sum()
            if gradNorm < 1.0e-8:
                self.dEnergy_dGrad_term1[i, j] = ti.Vector([0.0, 0.0])
                angle = ti.atan2(grad[1], grad[0])
                epsilons[i, j] = grad_energy_coef * (1.0 + aniso_magnitude * ti.cos(n_fold_symmetry * (angle - angle0)))
            else:
                angle = ti.atan2(grad[1], grad[0])
                epsilon = grad_energy_coef * (1.0 + aniso_magnitude * ti.cos(n_fold_symmetry * (angle - angle0)))
                epsilons[i, j] = epsilon
                dAngle_dGradX = -grad[1] / gradNorm
                dAngle_dGradY = grad[0] / gradNorm
                tmp = grad_energy_coef * aniso_magnitude * -ti.sin(n_fold_symmetry * (angle - angle0)) * n_fold_symmetry
                depsilon_dGrad = tmp * ti.Vector([dAngle_dGradX, dAngle_dGradY])
                self.dEnergy_dGrad_term1[i, j] = epsilon * depsilon_dGrad * gradNorm

        ### then, get the phi rate and temperature rate
        for i, j in phi:
            im, jm, ip, jp = self.neighbor_index(i, j)

            lapla_phi = (  # laplacian of phi
                2 * (phi[im, j] + phi[i, jm] + phi[ip, j] + phi[i, jp])
                + (phi[im, jm] + phi[im, jp] + phi[ip, jm] + phi[ip, jp])
                - 12 * phi[i, j]
            ) / (3.0 * dx * dx)
            lapla_tp = (  # laplacian of temperature
                2 * (temperature[im, j] + temperature[i, jm] + temperature[ip, j] + temperature[i, jp])
                + (temperature[im, jm] + temperature[im, jp] + temperature[ip, jm] + temperature[ip, jp])
                - 12 * temperature[i, j]
            ) / (3.0 * dx * dx)

            m_chem = self.alpha * ti.atan2(self.gamma * (self.temperature_equi - temperature[i, j]), 1.0)
            chemicalForce = phi[i, j] * (1.0 - phi[i, j]) * (phi[i, j] - 0.5 + m_chem)
            gradForce_term1 = self.divergence_dEnergy_dGrad_term1(i, j)
            grad_epsilon2 = ti.Vector(
                [
                    (epsilons[ip, j] ** 2 - epsilons[im, j] ** 2) / (2.0 * dx),
                    (epsilons[i, jp] ** 2 - epsilons[i, jm] ** 2) / (2.0 * dx),
                ]
            )
            grad_phi = ti.Vector(
                [
                    (phi[ip, j] - phi[im, j]) / (2.0 * dx),
                    (phi[i, jp] - phi[i, jm]) / (2.0 * dx),
                ]
            )
            gradForce_term2 = (
                grad_epsilon2[0] * grad_phi[0] + grad_epsilon2[1] * grad_phi[1] + epsilons[i, j] ** 2 * lapla_phi
            )

            self.phiRate[i, j][rk_loop] = mobility * (chemicalForce + gradForce_term1 + gradForce_term2)
            self.temperatureRate[i, j][rk_loop] = lapla_tp + self.latent_heat_coef * self.phiRate[i, j][rk_loop]

    @ti.kernel
    def rk4_total_update(
        self,
    ):
        """the final step in RK4 (Runge-Kutta) process"""
        (
            dt,
            phi,
            phi_old,
            temperature,
            temperature_old,
            phiRate,
            temperatureRate,
        ) = ti.static(
            self.dt,
            self.phi,
            self.phi_old,
            self.temperature,
            self.temperature_old,
            self.phiRate,
            self.temperatureRate,
        )
        for I in ti.grouped(phi):
            for k in ti.static(range(4)):
                phi_old[I] = phi_old[I] + self.weights_rk4[k] * dt * phiRate[I][k]
                temperature_old[I] = temperature_old[I] + self.weights_rk4[k] * dt * temperatureRate[I][k]
        for I in ti.grouped(phi):
            phi[I] = phi_old[I]
            temperature[I] = temperature_old[I]

    @ti.func
    def divergence_dEnergy_dGrad_term1(self, i, j):
        im, jm, ip, jp = self.neighbor_index(i, j)
        return (self.dEnergy_dGrad_term1[ip, j][0] - self.dEnergy_dGrad_term1[im, j][0]) / (2.0 * self.dx) + (
            self.dEnergy_dGrad_term1[i, jp][1] - self.dEnergy_dGrad_term1[i, jm][1]
        ) / (2.0 * self.dx)

    def advance(
        self,
    ):  # advance a time step
        self.get_rate(rk_loop=0)
        for rk_loop in range(1, 4):
            self.rk4_intermediate_update(rk_loop=rk_loop)
            self.get_rate(rk_loop=rk_loop)
        self.rk4_total_update()

    def getDendritic(self, steps=2048):
        self.initialize()
        gui = ti.GUI("phase field", res=(self.n, self.n))
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            for _ in range(self.showFrameFrequency):
                self.advance()

            gui.set_image(self.phi)
            gui.show()
        return self.phi


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f64)
    Dendrite().getDendritic(steps=10000)
