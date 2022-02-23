# %% ライブラリのインポート
from numba import jit
from numba import objmode
# from numba.experimental import jitclass
import numpy as np
import math
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
import time
# from multiprocessing import Pool

# FAVORITE COLORS (FAVOURITE COLOURS?)
color = ['#6667AB', '#0F4C81', '#5B6770', '#FF6F61', '#645394',
         '#84BD00', '#F6BE00', '#F7CAC9', '#16137E', '#45B8AC']


#
#
# %% SETTINGS FOR THE NEXT EXECUTION
date = '20220218_single2'
eV_array = 10000    # [eV]
alp = 1.0
pitchangle = 45  # degrees
colatitude = 80  # degrees
longitude = 10   # degrees
lam = 2.0        # degrees


#
#
# %% CONSTANTS
RJ = float(7E+7)        # Jupiter半径   単位: m
mJ = float(1.90E+27)    # Jupiter質量   単位: kg
RE = float(1.56E+6)     # Europa半径    単位: m
mE = float(4.8E+22)     # Europa質量    単位: kg

c = float(3E+8)         # 真空中光速    単位: m/s
me = float(9.1E-31)     # 電子質量      単位: kg
me = 2000*float(9.1E-31)     # 電子質量      単位: kg
e = float(-1.6E-19)     # 電子電荷      単位: C
e = float(1.6E-19)     # 電子電荷      単位: C

G = float(6.67E-11)     # 万有引力定数  単位: m^3 kg^-1 s^-2

mu = float(1.26E-6)     # 真空中透磁率  単位: N A^-2 = kg m s^-2 A^-2
Mdip = float(1.6E+27)   # Jupiterのダイポールモーメント 単位: A m^2
omgJ = float(1.74E-4)   # 木星の自転角速度 単位: rad/s
omgE = float(2.05E-5)   # Europaの公転角速度 単位: rad/s
omgR = omgJ-omgE        # 木星のEuropaに対する相対的な自転角速度 単位: rad/s
omgR = omgR*alp
eomg = np.array([-np.sin(np.radians(lam)),
                 0., np.cos(np.radians(lam))])
omgRvec = omgR*eomg


#
#
# %% 途中計算でよく出てくる定数の比
A1 = e/me                        # 運動方程式内の定数
A2 = (mu*Mdip)/(4*np.pi)         # ダイポール磁場表式内の定数
# A1 = float(-1.7582E+11)        # 運動方程式内の定数
# A2 = 1.60432E+20               # ダイポール磁場表式内の定数
A3 = 4*3.1415*me/(mu*Mdip*e)     # ドリフト速度の係数


#
#
# %% EUROPA POSITION (DETERMINED BY MAGNETIC LATITUDE)
# lam = 2.0  # =============== !!! ==============
L96 = 9.6*RJ  # Europa公転軌道 L値


# 木星とtrace座標系原点の距離(x軸の定義)
# Europaの中心を通る磁力線の脚(磁気赤道面)
R0 = L96*(np.cos(np.radians(lam)))**(-2)
R0x = R0
R0y = 0
R0z = 0
R0vec = np.array([R0x, R0y, R0z])


a1 = np.loadtxt(
    '/Users/shin/Documents/Research/Europa/Codes/gyrocenter/gyrocenter_2/gc301_50000ev_alp_1.0_20220218_single1.txt')
a2 = np.loadtxt(
    '/Users/shin/Documents/Research/Europa/Codes/gyrocenter/gyrocenter_2/gc301_50000ev_alp_1.0_20220218_single2.txt')


fig, ax = plt.subplots()
ax.set_aspect(1)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.plot((a1[:, 0]+R0vec[0])/RJ, (a1[:, 2]+R0vec[2])/RJ)
ax.plot((a2[:, 0]+R0vec[0])/RJ, (a2[:, 2]+R0vec[2])/RJ)
fig.tight_layout()
plt.show()
