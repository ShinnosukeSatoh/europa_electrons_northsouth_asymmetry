import numpy as np
from numba import jit, f8
from numba.experimental import jitclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
color = ['#0F4C81', '#FF6F61', '#645394', '#84BD00', '#F6BE00', '#F7CAC9']


@jit(nopython=True, fastmath=True)
def rk4(f, xv0, t, m, n):
    xv = np.zeros((m, n))
    xv[0, :] = xv0
    for k in range(m-1):
        tau = t[k+1] - t[k]
        F1 = f(xv[k, :], t[k])
        F2 = f(xv[k, :]+1/2*tau*F1, t[k]+1/2*tau)
        F3 = f(xv[k, :]+1/2*tau*F2, t[k]+1/2*tau)
        F4 = f(xv[k, :]+tau*F3, t[k]+tau)
        xv[k+1, :] = xv[k, :] \
            + tau/6*(F1 + 2*F2 + 2*F3 + F4)
    return xv


class RK4:
    def __init__(self, f, xv0, t):
        m = int(t.size)
        n = int(xv0.size)
        xv = rk4(f, xv0, t, m, n)
        self.x, self.y, self.z = xv[:, 0], xv[:, 1], xv[:, 2]
        self.vx, self.vy, self.vz = xv[:, 3], xv[:, 4], xv[:, 5]


# ダイポール磁場
@jit(nopython=True, fastmath=True)
def BB_dip(x):
    mu = 5E+10  # [H/m]
    A = mu/(4*3.14)
    # mum = -np.float((1.26*1.6/1.2)*1E+20)
    mum = -A
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    B_dipx = mum*3*x[2]*x[0]/r**5
    B_dipy = mum*3*x[2]*x[1]/r**5
    B_dipz = mum*((3*(x[2]/r)**2 - 1)/(r**3))
    B_dip = np.array([B_dipx, B_dipy, B_dipz])
    return B_dip


@jit(nopython=True, fastmath=True)
# 運動方程式
def f(x, t):
    mm = 1
    # A = q/mm
    A = 4E-4
    B = BB_dip(x)
    Bx = B[0]
    By = B[1]
    Bz = B[2]
    vdx = -x[1]*0.01
    vdy = x[0]*0.01
    f = np.array([x[3]+vdx, x[4]+vdy, x[5], A*(Ex+x[4]*Bz-x[5]*By),
                 A*(Ey+x[5]*Bx-x[3]*Bz), A*(Ez+x[3]*By-x[4]*Bx)])
    return f


def ax1(X, Y, Z):
    title = 'Charged Particle Tracing'

    # Figureを追加
    fig = plt.figure(figsize=plt.figaspect(1), dpi=100)

    # 3DAxesを追加
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1, 1, 1))

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Axesのタイトルを設定
    # ax.set_aspect(1)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-125, 125])
    ax.set_ylim([-125, 125])
    ax.set_zlim([-125, 125])
    ax.plot_surface(24*x+100, 24*y, 24*z,  rstride=4, cstride=4,
                    color=color[3], linewidth=0, alpha=0.9)
    ax.plot(X[::10], -Y[::10], Z[::10], color="red")
    # ax.legend()
    fig.tight_layout()

    plt.show()

    return 0


t = np.arange(0, 450, 1E-4)

E = np.array([0, 0, 0])
eE = np.array((np.linalg.norm(E)*E))
Ex, Ey, Ez = E

c = np.float(3E+8)
RJ = np.float(7E+1)
# x(0)=100, y(0)=100, z(0)=0, vx(0)=0, vy(0)=100, vz(0)=1
x0 = np.array([100, 0, 0, 0, 4.2, 2.5])
rk = RK4(f, x0, t)

x_min = -500
x_max = 500
y_min = -500
y_max = 500
z_min = -500
z_max = 500

# x_min= 0; x_max = 10.5
# y_min= -4; y_max = 4
# z_min= 0; z_max = 10.5

ax1(rk.x, rk.y, rk.z)

"""
plt.rcParams['agg.path.chunksize'] = 10000

fig, ax0 = plt.subplots(1, 2, dpi=(80), figsize=(12, 6))
ax0[0].set_xlabel('x')
ax0[0].set_ylabel('y')
ax0[0].set_title('x-y')
ax0[0].set_xlim(x_min, x_max)
ax0[0].set_ylim(y_min, y_max)
ax0[0].plot(rk.x, rk.y, label='x-y', color=color[0], linewidth=0.5)
ax0[0].legend(loc='upper left')

ax0[1].set_xlabel('x')
ax0[1].set_ylabel('z')
ax0[1].set_title('x-z')
ax0[1].set_xlim(x_min, x_max)
ax0[1].set_ylim(z_min, z_max)
ax0[1].plot(rk.x, rk.z, label='x-z', color=color[0], linewidth=0.5)
# ax0[1].scatter(rk.x[0], rk.y[0], rk.z[0], color
# =color[1])
ax0[1].legend(loc='upper left')
fig.tight_layout()
plt.show()
"""
