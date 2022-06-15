""" gc_304_plot.py

Created on Wed Jun 15 2022
@author: Shin Satoh

Description:
Previous programs for plotting (e.g. gc_303_plot.py) include
`np.histogram2d` function, which enables a fast calculation of
particle number flux [cm-2 s-1]. However, the method cannot
derive the differential flux [cm-2 s-1 str-1 eV-1].
This program is intended to calculate differential flux without
using np.histogram2d method.

Version:
1.0.0 (Jun 15, 2022)

"""


# %% ライブラリのインポート
from statistics import mode
from numba import jit
# from numba.experimental import jitclass
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
from matplotlib.colors import LinearSegmentedColormap  # colormapをカスタマイズする
# from matplotlib import rc
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
# import time

# from numpy.lib.npyio import savez_compressed
# from multiprocessing import Pool

# from numpy.lib.function_base import _flip_dispatcher
color = ['#0F4C81', '#FF6F61', '#645394',
         '#84BD00', '#F6BE00', '#F7CAC9', '#0F80E6']


# matplotlib フォント設定
plt.rcParams.update({'font.sans-serif': "Arial",
                     'font.family': "sans-serif",
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:italic:bold'
                     })


#
#
# %% TOGGLE
MOON = 'EUROPA'            # IO, EUROPA, GANYMEDE
FORWARD_BACKWARD = -1  # 1=FORWARD, -1=BACKWARD
GYRO = 1               # 0=GUIDING CENTER, 1=GYRO MOTION
ION_ELECTRON = 0       # 0=ELECTRON, 1=ION
Z = -1                 # CHARGE (-1 for ELECTRON)
U = 32                 # ATOMIC MASS (32 for SULFUR)
kBT1 = 20              # [EUROPA] CENTER TEMPERATURE [eV] 95%
kBT2 = 300             # [EUROPA] CENTER TEMPERATURE [eV] 5%
# kBT1 = 5               # [IO] CENTER TEMPERATURE [eV] 95%
# kBT2 = 5               # [IO] CENTER TEMPERATURE [eV] 5%
ne = 160               # ELECTRON DENSITY 160 cm-3, S++ DENSITY 20 cm-3
# ne = 2500              # [IO] ELECTRON DENSITY [cm-3]
nc_O2 = 1.4E+15        # O2 COLUMN DENSITY [cm-2]
# nc_O2 = 1.0E+16      # [IO] O2 COLUMN DENSITY [cm-2]


#
#
# %% SETTINGS FOR THE NEXT EXECUTION
code = 'gc304'
date = '20220608e_'+str(MOON)

# これはgc303用
eV_array = np.array([
    2, 4, 6, 8, 10, 12,
    14, 16, 18, 20,                 # Cross sectionあり
    25, 30, 40, 50, 60, 80, 100,    # Cross sectionあり
    200, 300, 400, 500, 600,        # Cross sectionあり
    800, 1000,
    2500, 5000, 7500, 10000,
    # 25000, 50000, 75000
])    # [eV]

alp = 0.05
lam = 10.0        # degrees
# 'FLUX'...precipitation flux, '1356'...brightness, 'ENERGY'...energy flux
MODE = 'FLUX'
pitch = 45        # ピッチアングル分割数
long = int(60)    # 経度分割数
lat = int(long/2)  # 緯度分割数
save_yn = 0       # save? y=1, n=0

# 励起断面積(14eVから600eVまで)
# これはgc303用
crosssections = np.array([
    0, 0, 0, 0, 0, 0,
    0.0003, 1.68, 2.73, 3.37, 4.11, 4.18, 4.72, 5.74,
    6.45, 6.76, 6.40, 4.79, 3.79, 3.13, 2.62, 2.21,
    0, 0,
    0, 0, 0, 0,
    # 0, 0, 0
])

crosssections = np.array([
    0, 0, 0, 0, 0, 0,
    0.0003, 1.68, 2.73, 3.37, 4.11, 4.18, 4.72, 5.74,
    6.45, 6.76, 6.40, 4.79, 3.79, 3.13, 2.62, 2.21,
    2, 2,
    2, 2, 2, 2,
    # 2, 2, 2
])

# エネルギー一覧
enlist = eV_array
devlist = enlist[1:] - enlist[:-1]
devlist = np.append(devlist, devlist[-1]*2)
devlist = np.array([
    2, 2, 2, 2, 2, 2,
    2, 2, 2, 5,                 # Cross sectionあり
    5, 5, 10, 10, 20, 20, 100,    # Cross sectionあり
    100, 100, 100, 100, 200,        # Cross sectionあり
    200, 1500,
    2500, 2500, 2500, 15000,
    # 25000, 25000, 25000
])    # [eV]


#
#
# %% CONSTANTS
RJ = 7E+7               # Jupiter半径   単位: m
mJ = 1.90E+27           # Jupiter質量   単位: kg

if MOON == 'IO':
    RE = 1.82E+6        # 衛星半径    単位: m
    mE = 8.94E+22       # 衛星質量    単位: kg
    L94 = 5.9*RJ        # 衛星と木星中心の距離 単位: km
    omgE = 4.1105E-5    # 衛星の公転角速度 単位: rad s^-1
elif MOON == 'EUROPA':
    RE = 1.56E+6        # 衛星半径    単位: m
    mE = 4.8E+22        # 衛星質量    単位: kg
    L94 = 9.4*RJ        # 衛星と木星中心の距離 単位: km
    omgE = 2.0478E-5    # 衛星の公転角速度 単位: rad s^-1

c = float(3E+8)         # 真空中光速    単位: m/s
G = float(6.67E-11)     # 万有引力定数  単位: m^3 kg^-1 s^-2

NA = 6.02E+23           # アボガドロ数
me = float(9.1E-31)     # 電子質量   単位: kg
if ION_ELECTRON == 1:
    me = me*1836*U      # 荷電粒子質量 単位: kg
    print(me)
e = Z*float(1.6E-19)    # 電荷 単位: C

mu = float(1.26E-6)     # 真空中透磁率  単位: N A^-2 = kg m s^-2 A^-2
Mdip = float(1.6E+27)   # Jupiterのダイポールモーメント 単位: A m^2
omgJ = float(1.74E-4)   # 木星の自転角速度 単位: rad/s
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


# 木星とtrace座標系原点の距離(x軸の定義)
R0 = L94*(np.cos(np.radians(lam)))**(-2)
R0x = R0
R0y = 0
R0z = 0

# Europaのtrace座標系における位置
eurx = L94*math.cos(math.radians(lam)) - R0x
eury = 0 - R0y
eurz = L94*math.sin(math.radians(lam)) - R0z


# ========== UNDER CONSTRUCTION BELOW ==========
