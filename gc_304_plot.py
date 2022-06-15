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
MOON = 'EUROPA'         # IO, EUROPA, GANYMEDE
FORWARD_BACKWARD = -1   # 1=FORWARD, -1=BACKWARD
GYRO = 1                # 0=GUIDING CENTER, 1=GYRO MOTION
ION_ELECTRON = 0        # 0=ELECTRON, 1=ION
Z = -1                  # CHARGE (-1 for ELECTRON)
U = 32                  # ATOMIC MASS (32 for SULFUR)
kBT1 = 20               # [EUROPA] CENTER TEMPERATURE [eV] 95%
kBT2 = 300              # [EUROPA] CENTER TEMPERATURE [eV] 5%
# kBT1 = 5              # [IO] CENTER TEMPERATURE [eV] 95%
# kBT2 = 5              # [IO] CENTER TEMPERATURE [eV] 5%
ne = 160                # ELECTRON DENSITY 160 cm-3, S++ DENSITY 20 cm-3
# ne = 2500             # [IO] ELECTRON DENSITY [cm-3]
nc_O2 = 1.4E+15         # O2 COLUMN DENSITY [cm-2]
# nc_O2 = 1.0E+16       # [IO] O2 COLUMN DENSITY [cm-2]


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


#
#
# %% マクスウェル速度分布関数
@jit(nopython=True, fastmath=True)
def maxwell(en):
    # en: 電子のエネルギー [eV]
    v = np.sqrt((en/me)*2*float(1.602E-19))

    # 中心 20eV
    kBT = kBT1  # eV
    kBT = kBT*(1.6E-19)  # J
    fv20 = 4*np.pi*(v**2) * (me/(2*np.pi*kBT))**(1.5) * \
        np.exp(-(me*v**2)/(2*kBT))

    # 中心 300eV
    kBT = kBT2  # eV
    kBT = kBT*(1.6E-19)  # J
    fv300 = 4*np.pi*(v**2) * (me/(2*np.pi*kBT))**(1.5) * \
        np.exp(-(me*v**2)/(2*kBT))

    fv = 0.95*fv20 + 0.05*fv300

    return fv


#
#
# %%
def dataload(filepath):
    # 座標&ピッチ角ファイル
    a0 = np.loadtxt(filepath)
    # a0[:, 0] ... 出発点 x座標
    # a0[:, 1] ... 出発点 y座標
    # a0[:, 2] ... 出発点 z座標
    # a0[:, 3] ... 終点 x座標
    # a0[:, 4] ... 終点 y座標
    # a0[:, 5] ... 終点 z座標
    # a0[:, 6] ... yn (=gridnum)
    # a0[:, 7] ... 終点 energy [eV]
    # a0[:, 8] ... 終点 alpha_eq [RADIANS]
    # a0[:, 9] ... 出発点 v_dot_n

    # nan除外
    a0 = a0[np.where(~np.isnan(a0).any(axis=1))]

    if MODE == 'FLUX':
        # vdotn < 0 のみが適切
        a0 = a0[np.where(a0[:, 9] < 0)]
        # print('vdotn: ', a0[:, 9])

    # 表面着地点座標 & ピッチ角 (検索は表面y座標=1列目)
    xyz = a0[:, 0:3]
    gridnum = a0[:, 6]
    energy = a0[:, 7]
    aeq = a0[:, 8]
    vdotn = a0[:, 9]

    return xyz, energy, aeq, vdotn, gridnum


# ヒストグラム初期化
H1d = np.zeros(long*lat)
H2d = np.zeros((lat, long))
H3d = np.zeros((len(enlist), lat-1, long-1))
gamma3d = np.zeros((len(enlist), lat-1, long-1))


# ヒストグラム積分(v方向)
for i in range(len(enlist)):
    # ヒストグラム初期化
    H2di = np.zeros((lat, long))

    Ei = enlist[i]      # i番目のエネルギー [eV]
    dEi = devlist[i]    # i番目からi+1番目のエネルギー差 [eV]

    filepath0 = str(code)+'_'+str(date)+'_' + \
        str(Ei)+'ev_alp_'+str(alp)+'.txt'
    xyz0, energy0, aeq0, vdotn0, gridnum0 = dataload(filepath0)

    # 速度分布関数
    fv2 = maxwell(Ei)
    v1 = np.sqrt((Ei+dEi)*2*(1.6E-19)/me)
    v2 = np.sqrt((Ei)*2*(1.6E-19)/me)
    dv = v1 - v2

    # 速度ベクトルと法線ベクトルがなす角(theta_s)
    theta_s = np.arccos(vdotn0/v1)

    # 表面個数を数えあげ
    # j番目の格子点で数え上げる
    for j in range(long*lat):
        j_index = np.where(gridnum0 == j)   # j番目格子点のインデックス

        # 速度ベクトル関連
        vdotn1 = -vdotn0[j_index]     # 速度ベクトルと法線ベクトルの内積 [m/s]
        vdotn1_sum = np.sum(vdotn1)  # 内積の和 [m/s]
        vdotn1_sum *= 100            # 内積の和 [cm/s]

        # 立体角関連
        theta_si = theta_s[j_index]  # 速度ベクトルと法線ベクトルがなす角 [rad]
        if theta_si.size > 0:  # NOT EMPTY
            omg_s = 2*np.max(theta_si)   # j番目格子点に入射することが可能な立体角 [rad]
        else:
            omg_s = -1

        # ピッチ角関連
        fa2 = 1/np.pi       # ピッチ角分布
        fa2 = np.sin(aeq0[j_index])**0.5
        da = np.pi/pitch    # ピッチ角刻み

        # j番目格子点: エネルギー Ei~Ei+dEi [eV] の個数フラックス [cm-2 s-1]
        num_flux_j = np.sum(vdotn1*ne*fv2*fa2*dv*da)

        # j番目格子点: エネルギー Ei~Ei+dEi [eV] の電子数密度 [cm-3]
        density_j = np.sum(np.ones(vdotn1.shape)*ne*fv2*fa2*dv*da)

        # j番目格子点: エネルギー Ei~Ei+dEi [eV] のdifferential flux [cm-2 s-1 str-1 eV-1]
        diff_j = np.sum(vdotn1*ne*fv2*fa2*dv*da/(omg_s*Ei))

        H1d[j] = diff_j

    # 緯度経度マップにreshape
    H2di = H1d.reshape(lat, long)

    H2d += H2di

# H2dは横軸(経度方向)を正しく入れ替える必要あり
# 現状 ... 180W -> 90W -> 0W/360W -> 270W -> 180W
# これを ... 360W -> 270W -> 180W -> 90W -> 0W に直す
H2d = np.append(H2d[:, int(long/2):], H2d[:, 0:int(long/2)], axis=1)  # 変換した

ns_ratio = np.average(H2d[0:int(lat/2), :]/H2d[int(lat/2):, :])
print(ns_ratio)

# テストプロット用
x = np.linspace(0, 360, long+1)
y = np.linspace(0, 180, lat+1)
X, Y = np.meshgrid(x, y)
print(X.shape)

fig, ax = plt.subplots()
ax.invert_yaxis()
mappable0 = ax.pcolormesh(X, Y, H2d)
pp = fig.colorbar(mappable0)

plt.savefig('test_H2d_20220615.png')

plt.show()


# ========== UNDER CONSTRUCTION BELOW ==========
