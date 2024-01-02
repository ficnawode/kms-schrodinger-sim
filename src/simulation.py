import numpy as np
from scipy import linalg
from numba import njit
from matplotlib import pyplot as plt
from matplotlib import animation
import tqdm


class SchrodingerSim():
    def __init__(self, V: callable, dx=0.05, x_min=-5, x_max=50, dt=0.002, J=2000, k_0=10, dk=0.5, show_plots=False):
        self.dx = dx
        self.x = np.arange(x_min, x_max + dx, dx)
        self.dt = dt
        self.J = J
        self.k_0 = k_0
        self.dk = dk
        self.show_plots = show_plots
        self.V = V

    @staticmethod
    def psi_0(x, dk, k_0):
        return np.sqrt(dk)/(np.pi**(1/4))*np.exp(-x**2*dk**2/2)*np.exp(1j*k_0*x)

    @staticmethod
    def tridiag(a, b, c, k1=-1, k2=0, k3=1):
        return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

    def calculate_r_vec(self, psi, V):
        def r_vec_i(psi_jm1, psi_j, psi_jp1, V_j):
            return psi_j + 1j*self.dt/2 * ((psi_jp1 - 2*psi_j + psi_jm1)/(self.dx**2) - V_j*psi_j)

        return [r_vec_i(p_jm1, p_j, p_jp1, V_j) for p_jm1, p_j, p_jp1, V_j in zip(psi[:-2], psi[1:-1], psi[2:], V[1:-1])]

    def make_animation(self, gif_path: str):
        psi_x_0 = self.psi_0(self.x, self.dk, self.k_0)
        r_vec = self.calculate_r_vec(psi_x_0, self.V(self.x))
        b_diag = 1 + 1j*self.dt/2*(2/(self.dx**2) + self.V(self.x)[1:-1])
        a_c_diag = -1j*self.dt/(2*self.dx**2)*np.ones(len(b_diag) - 1)
        A_mat = self.tridiag(a_c_diag, b_diag, a_c_diag)

        psi_i = self.psi_0(self.x, self.dk, self.k_0)
        psi_s = []
        for i in tqdm.tqdm(range(self.J)):
            psi_i = np.linalg.solve(A_mat, r_vec)
            psi_s.append([0, *psi_i, 0])
            r_vec = self.calculate_r_vec([0, *psi_i, 0], self.V(self.x))

        def animate(i):
            print(i)
            ln1.set_ydata(np.array(np.absolute(psi_s[i])**2))
            return ln1,

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ln1, = ax.plot(self.x, np.absolute(self.psi_0(self.x, self.dk, self.k_0))
                       ** 2, 'r-', lw=2, markersize=8)
        ln2 = ax.plot(self.x, self.V(self.x)/100)
        ax.set_ylim(-1, 2)
        ax.set_xlim(-5, 50)
        ax.set_ylabel(r'$|\psi(x)|^2$')
        ax.set_xlabel(r'$x/L$')
        plt.tight_layout()
        ani = animation.FuncAnimation(fig, animate, frames=self.J, interval=1)
        ani.save(gif_path, writer='pillow', fps=30, dpi=200)
