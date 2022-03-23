""" gc_303_plot.py

Created on Thu Mar 17 2022
@author: Shin Satoh

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
FORWARD_BACKWARD = -1  # 1=FORWARD, -1=BACKWARD
GYRO = 1               # 0=GUIDING CENTER, 1=GYRO MOTION
ION_ELECTRON = 0       # 0=ELECTRON, 1=ION
Z = -1                 # CHARGE (-1 for ELECTRON)
U = 32                 # ATOMIC MASS (32 for SULFUR)
kBT1 = 20              # CENTER TEMPERATURE [eV] 95%
kBT2 = 300             # CENTER TEMPERATURE [eV] 5%
ne = 160               # ELECTRON DENSITY 160 cm-3, S++ DENSITY 20 cm-3
nc_O2 = 1.4E+15        # O2 COLUMN DENSITY [cm-2]


#
#
# %% SETTINGS FOR THE NEXT EXECUTION
code = 'gc303'
date = '20220322e_94_3'

# これはgc303用
eV_array = np.array([
    2, 4, 6, 8, 10, 12,
    14, 16, 18, 20,                 # Cross sectionあり
    25, 30, 40, 50, 60, 80, 100,    # Cross sectionあり
    200, 300, 400, 500, 600,        # Cross sectionあり
    800, 1000,
    2500, 5000, 7500, 10000
])    # [eV]
"""
# これはgc301用
eV_array = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    15, 20,                         # Cross sectionあり
    25, 30, 40, 50, 60, 80, 100,    # Cross sectionあり
    200, 300, 400, 500, 700,        # Cross sectionあり
    1000, 2000, 3000, 4000, 5000,
    7000, 10000, 20000
])    # [eV]
"""

alp = 0.025
lam = 10.0        # degrees
MODE = '1356'     # 'FLUX'...precipitation flux, '1356'...brightness
save_yn = 1       # save? y=1, n=0

# 励起断面積(14eVから600eVまで)
# これはgc303用
crosssections = np.array([
    0, 0, 0, 0, 0, 0,
    0.0003, 1.68, 2.73, 3.37, 4.11, 4.18, 4.72, 5.74,
    6.45, 6.76, 6.40, 4.79, 3.79, 3.13, 2.62, 2.21,
    0, 0,
    0, 0, 0, 0
])
"""
# これはgc301用
crosssections = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.0003, 3.37,
    4.11, 4.18, 4.72, 5.74, 6.45, 6.76, 6.40,
    4.79, 3.79, 3.13, 2.62, 2.21,
    0, 0, 0, 0, 0,
    0, 0, 0
])
"""
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
    2500, 2500, 2500, 15000
])    # [eV]


#
#
# %% CONSTANTS
RJ = float(7E+7)        # Jupiter半径   単位: m
mJ = float(1.90E+27)    # Jupiter質量   単位: kg
RE = float(1.56E+6)     # Europa半径    単位: m
mE = float(4.8E+22)     # Europa質量    単位: kg

c = float(3E+8)         # 真空中光速    単位: m/s
G = float(6.67E-11)     # 万有引力定数  単位: m^3 kg^-1 s^-2

NA = 6.02E+23           # アボガドロ数
me = float(9.1E-31)     # 電子質量   単位: kg
if ION_ELECTRON == 1:
    me = me*1836*U      # 荷電粒子質量 単位: kg
e = Z*float(1.6E-19)    # 電荷      単位: C

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


# %% Europa Position
# lam = 10.0
REr = 9.4*RJ  # Europa公転軌道 L値  ======= !!!!! ========

# 木星とtrace座標系原点の距離(x軸の定義)
R0 = REr*(np.cos(np.radians(lam)))**(-2)
R0x = R0
R0y = 0
R0z = 0

# Europaのtrace座標系における位置
eurx = REr*math.cos(math.radians(lam)) - R0
eury = 0
eurz = REr*math.sin(math.radians(lam))


#
#
# %% 励起断面積フィッティング関数
def Qx(ev):
    E = 0.968640893964833
    Eij = 14.26                    # eV
    X = ev/Eij
    C0 = 0.48000650E-01
    C1 = -0.77856300E-01
    C2 = -0.39284400E-01
    C3 = 0
    C4 = -0.53260800
    C5 = 0.47435400
    C6 = -0.47435400
    C7 = 0.16287750
    C8 = 0.16596000

    OMGx_1 = (C0*(1-(1/X))/(X**2) + C1*(X-1)*np.exp(-C8*X) + C2*(X-1)*np.exp(-2*C8*X) +
              C3*(X-1)*np.exp(-3*C8*X) + C4*(X-1)*np.exp(-4*C8*X) + C5 + C6/X + C7*np.log(X))
    Qx_1 = OMGx_1/(E*X)
    Qx_1 = Qx_1*(8.8E-17)

    Eij = 37.50                    # eV
    X = ev/Eij
    C0 = 0.25263500E-02
    C1 = -0.40977000E-02
    C2 = -0.20676000E-02
    C3 = 0
    C4 = 0.28032000E-01
    C5 = -0.24966000E-01
    C6 = -0.24966000E-01
    C7 = 0.85725000E-02
    C8 = 0.16596000

    OMGx_2 = (C0*(1-(1/X))/(X**2) + C1*(X-1)*np.exp(-C8*X) + C2*(X-1)*np.exp(-2*C8*X) +
              C3*(X-1)*np.exp(-3*C8*X) + C4*(X-1)*np.exp(-4*C8*X) + C5 + C6/X + C7*np.log(X))

    Qx_2 = OMGx_2/(E*X)
    Qx_2 = Qx_2*(8.8E-17)

    Qx = 0.95*Qx_1+0.05*+Qx_2

    Qx[np.where(ev < 14.26)] = 0

    return Qx


#
#
# %% 磁場
@jit('Tuple((f8,f8,f8))(f8,f8,f8)', nopython=True, fastmath=True)
def Bfield(x, y, z):
    # x, y, zは木星からの距離
    R2 = x**2 + y**2
    r_5 = math.sqrt(R2 + z**2)**(-5)

    Bx = A2*(3*z*x*r_5)
    By = A2*(3*z*y*r_5)
    Bz = A2*(2*z**2 - R2)*r_5

    return Bx, By, Bz


#
#
# %%
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
# %% 着地点における速度ベクトル(x, y, z成分)
def V(xyza, veq):
    # 木星原点に変換
    x = xyza[:, 0] + R0x
    y = xyza[:, 1] + R0y
    z = xyza[:, 2] + R0z
    aeq = xyza[:, 3]

    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)

    # Parallel
    nakami = 1 - r**5 * np.sin(aeq)**2 * \
        np.sqrt(r**2 + 3 * z**2) * (x**2 + y**2)**(-3)

    coef = veq * np.sqrt(nakami) * (r**(-1)) * (r**2 + 3 * z**2)**(-0.5)

    v_p_x = 3 * z * x
    v_p_y = 3 * z * y
    v_p_z = (2 * z**2) - (x**2 + y**2)

    v_par = np.stack([coef * v_p_x, coef * v_p_y, coef * v_p_z], 1)
    vpa2 = (coef**2) * (v_p_x**2 + v_p_y**2 + v_p_z**2)

    # Drift
    B = A2 * (np.sqrt(1+3*(z/r))**2) * r**(-3)
    Beq = A2 * np.sqrt(R0x**2 + R0y**2 + R0z**2)**(-3)
    vpe2 = (veq**2) * (B/Beq) * np.sin(aeq)**2
    theta = np.arccos(z/r)
    nakami = (vpa2) * r * np.sqrt(x**2 + y**2)
    nakami2 = 0.5 * vpe2 * (r**2) * np.sin(theta) * \
        (1 + (z/r)**2) * (1 + 3*(z/r)**2)**(-1)

    vb = A3 * (1+3*(z/r)**2)**(-1) * (nakami+nakami2)
    vb_x = np.zeros(x.shape)
    vb_y = omgR*x + vb
    vb_z = np.zeros(vb_x.shape)

    v_drift = np.stack((vb_x, vb_y, vb_z), 1)

    return v_par + v_drift


#
#
# %% 2つの3次元ベクトルの内積から、なす角を計算
@jit(nopython=True, fastmath=True)
def angle(A, B):
    # A, B... 3次元ベクトル

    # AベクトルとBベクトルの内積
    Dot = (A[:, 0]*B[:, 0] + A[:, 1]*B[:, 1] + A[:, 2]*B[:, 2])/(np.sqrt(A[:, 0] **
                                                                         2 + A[:, 1]**2 + A[:, 2]**2) * np.sqrt(B[:, 0]**2 + B[:, 1]**2 + B[:, 2]**2))

    # なす角
    ang = np.arccos(Dot)    # 単位: RADIANS

    return ang


#
#
# %% z軸周りの回転
@jit(nopython=True, fastmath=True)
def rot_xy(x, y, z, dp):
    xrot = x*np.cos(dp) + y*np.sin(dp)
    yrot = -x*np.sin(dp) + y*np.cos(dp)
    zrot = z
    rot = np.stack((xrot, yrot, zrot), axis=1)

    return rot


#
#
# %% 座標変換(Europa軌道座標系から木星dipole座標系に)
@jit(nopython=True, fastmath=True)
def rot_dipole(xyzad, lam):
    xi = xyzad[:, 0]
    xj = xyzad[:, 1]
    xk = xyzad[:, 2]

    xrot = xi*np.cos(np.radians(lam))+xk*np.sin(np.radians(lam))
    yrot = xj
    zrot = -xi*np.sin(np.radians(lam))+xk*np.cos(np.radians(lam))

    rot = np.stack((xrot, yrot, zrot, xyzad[:, 3], xyzad[:, 4]), axis=1)
    return rot


#
#
# %% 座標変換(Europa軌道座標系から木星dipole座標系に)
@jit(nopython=True, fastmath=True)
def rot_dipole2(xyz, lam):
    xi = xyz[0]
    xj = xyz[1]
    xk = xyz[2]

    xrot = xi*np.cos(np.radians(lam))+xk*np.sin(np.radians(lam))
    yrot = xj
    zrot = -xi*np.sin(np.radians(lam))+xk*np.cos(np.radians(lam))

    rot = np.array([xrot, yrot, zrot])
    return rot


#
#
# %% Europa中心座標に変換
@jit(nopython=True, fastmath=True)
def Europacentric(xyz, aeq):
    xyz[:, 0] += R0   # x座標原点を木星に
    rot = rot_dipole(xyz, lam)
    eur_rot = rot_dipole2(np.array([eurx+R0, eury, eurz]), lam)
    rot += - eur_rot  # Europa中心に原点を置き直す
    rxyza = np.stack((rot[:, 0], rot[:, 1], rot[:, 2], aeq), 1)   # aeq含む
    return rxyza


#
#
# %% 単位面積あたりのフラックスに直す
@jit(nopython=True, parallel=True)
def perarea(Xmesh, Ymesh, H):
    ntheta = int(300)
    dtheta = np.radians((Ymesh[1, 0]-Ymesh[0, 0])/ntheta)
    theta = np.radians(Ymesh[:-1, :-1])  # thetaのスタート

    nphi = int(300)
    dphi = np.radians((Xmesh[0, 1]-Xmesh[0, 0])/nphi)

    s = np.zeros(H.shape)  # ビンの面積
    for i in range(ntheta):
        s0 = np.sin(theta)
        theta += dtheta
        s1 = np.sin(theta)
        s += 0.5*(s0+s1)*dtheta

    for j in range(nphi):
        s += dphi

    # s = s * REU**2

    # 緯度経度マップ上のグリッドで規格化する
    # s = dtheta * dphi

    # 単位面積あたりにする[m^-2]
    # H = H*s

    return H


#
#
# %% 着地点の余緯度と経度を調べる
@jit
def find(xyz0, aeq, energy, vdotn):
    # 電子の座標(Europa centric)
    # x... anti jovian
    # y... corotation
    # z... northern

    x = xyz0[:, 0]
    y = xyz0[:, 1]
    z = xyz0[:, 2]
    theta = np.arccos(z/np.sqrt(x**2+y**2+z**2))
    phi = np.arctan2(y, x)

    # phi... -pi to pi [RADIANS]
    # theta... 0 to pi [RADIANS]

    maparray = np.stack((phi, theta), 1)

    # ヒストグラムの作成
    # k = int(1 + np.log2(21*216000))
    maparray = np.degrees(maparray)
    xedges = list(np.linspace(-180, 180, 80))
    yedges = list(np.linspace(0, 180, 40))

    # 'FLUX'モードの時は重み付けあり
    if MODE == 'FLUX':
        w = np.abs(vdotn)  # m s-1
        w = w * 100        # cm s-1
        H, xedges, yedges = np.histogram2d(
            maparray[:, 0], maparray[:, 1], bins=(xedges, yedges),
            weights=w
        )
        H = H.T

    # '1356'モードの時は重み付けなし
    if MODE == '1356':
        H, xedges, yedges = np.histogram2d(
            maparray[:, 0], maparray[:, 1], bins=(xedges, yedges),
        )
        H = H.T

    # ピッチ角分布
    f_dalpha = 1/pitch
    H *= f_dalpha

    # メッシュの作成
    X, Y = np.meshgrid(xedges, yedges)

    return H, X, Y


#
#
# %% マップ作成
def mapplot(H, X, Y):
    # x軸ラベル(標準的なwest longitude)
    xticklabels = ['360$^\\circ$\nW', '270$^\\circ$W',
                   '180$^\\circ$W', '90$^\\circ$W', '0$^\\circ$']
    yticklabels = ['90$^\\circ$N', '45$^\\circ$N',
                   '0$^\\circ$', '45$^\\circ$S', '90$^\\circ$S']

    # 図のタイトル
    title = 'Latitude-West Longitude, $\\lambda=$' + \
        str(lam) + '$^\\circ$'

    # 描画
    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.set_aspect(1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('West Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_xticks(np.linspace(-180, 180, 5))
    ax.set_yticks(np.linspace(0, 180, 5))
    ax.set_xticklabels(xticklabels, fontsize=12)
    ax.set_yticklabels(yticklabels, fontsize=12)
    ax.text(3, 8, 'boxed italics text in data coords', style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax.invert_yaxis()
    mappable0 = ax.pcolormesh(X, Y, H, cmap='magma', vmin=0)
    pp = fig.colorbar(mappable0, orientation='vertical')
    if MODE == 'FLUX':
        pp.set_label('Precipitation Flux [cm$^{-2}$ s$^{-1}$]', fontsize=12)
    if MODE == '1356':
        pp.set_label('135.6 nm Brightness [Rayleigh]', fontsize=12)
    pp.ax.tick_params(labelsize=12)
    pp.ax.yaxis.get_offset_text().set_fontsize(12)
    fig.tight_layout()

    if save_yn == 1:
        plt.savefig('P'+str(code)+'_'+str(date) +
                    '_alp_'+str(alp)+'_'+str(MODE)+'_2.png')

    plt.show()
    # plt.close()

    return 0


#
#
# %% オリジナルのカラーマップ
def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


#
#
# %% マップ作成
def mapplot2(H, X, Y):
    plt.rcParams['figure.subplot.bottom'] = 0.18
    plt.rcParams['figure.subplot.right'] = 0.83

    # x軸ラベル(標準的なwest longitude)
    xticklabels = ['360 W', '270 W',
                   '180 W', '90 W', '0']
    yticklabels = ['-90', '-45',
                   '0', '45', '90']

    # 図のタイトル
    if MODE == 'FLUX':
        title = 'Electron Precipitation Flux, $\\lambda=$' + \
            str(lam) + '$^\\circ$'

    # 図のタイトル
    if MODE == '1356':
        title = 'OI 135.6 nm Brightness, $\\lambda=$' + \
            str(lam) + '$^\\circ$'

    # 描画
    fig, ax = plt.subplots(
        2, 2,
        dpi=150,
        figsize=(8, 4),
        gridspec_kw={'width_ratios': [1, 18], 'height_ratios': [9, 1]}
    )

    # 左上 [0,0]
    ax[0, 0].set_ylabel('Latitude [degrees]', fontsize=15)  # y軸のタイトル
    ax[0, 0].set_yticks(np.linspace(0, 180, 5))
    ax[0, 0].set_yticklabels(yticklabels, fontsize=15)   # y軸のticks
    ax[0, 0].set_facecolor('#555555')    # グレーで塗りつぶす

    # 右上 [0,1]
    ax[0, 1].set_title(title, fontsize=15, weight='bold')    # 図のタイトル
    ax[0, 1].invert_yaxis()
    if MODE == 'FLUX':
        mappable0 = ax[0, 1].pcolormesh(X, Y, H,
                                        cmap='magma',
                                        vmin=0
                                        )
    if MODE == '1356':
        mappable0 = ax[0, 1].pcolormesh(X, Y, H,
                                        cmap=generate_cmap(['#000000',
                                                            '#010E5E',
                                                            '#042AA6',
                                                            '#0F7CE0',
                                                            '#1AC7FF',
                                                            '#EEFBFF'
                                                            ]),
                                        vmin=0
                                        )

    # 左下 [1,0]
    # ax[1, 0].set_facecolor('#FFFFFF')  # 黒で塗りつぶす

    # 右下 [1,1]
    ax[1, 1].set_xlabel('Longitude [degrees]', fontsize=15)    # x軸のタイトル
    ax[1, 1].set_xticks(np.linspace(-180, 180, 5))
    ax[1, 1].set_xticklabels(xticklabels, fontsize=15)   # x軸のticks
    ax[1, 1].set_facecolor('#555555')    # グレーで塗りつぶす

    # [0,0]のグラフのx軸の表記を消去
    ax[0, 0].tick_params('x', length=0, which='major')  # 目盛りを消す
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)  # ラベルを消す

    # [0,1]のグラフの軸の表記を全て消去
    ax[0, 1].tick_params('x', length=0, which='major')  # 目盛りを消す
    ax[0, 1].tick_params('y', length=0, which='major')  # 目盛りを消す
    plt.setp(ax[0, 1].get_xticklabels(), visible=False)  # ラベルを消す
    plt.setp(ax[0, 1].get_yticklabels(), visible=False)  # ラベルを消す

    # [1,0]のグラフの軸の表記を全て消去
    ax[1, 0].tick_params('x', length=0, which='major')  # 目盛りを消す
    ax[1, 0].tick_params('y', length=0, which='major')  # 目盛りを消す
    plt.setp(ax[1, 0].get_xticklabels(), visible=False)  # ラベルを消す
    plt.setp(ax[1, 0].get_yticklabels(), visible=False)  # ラベルを消す
    ax[1, 0].axis("off")                                # 全部消す

    # 下のグラフのy軸の表記を消去
    ax[1, 1].tick_params('y', length=0, which='major')  # 目盛りを消す
    plt.setp(ax[1, 1].get_yticklabels(), visible=False)  # ラベルを消す

    # 4枚を結合
    plt.subplots_adjust(wspace=.0)
    plt.subplots_adjust(hspace=.0)

    # 先行後行半球
    ax[1, 1].axvline(x=0, color='#000000')
    ax[1, 1].text(-111, 0.25,
                  'Trailing',
                  color='#FFFFFF',
                  weight='bold',
                  # style='italic',
                  fontsize=14)
    ax[1, 1].text(68, 0.25,
                  'Leading',
                  color='#FFFFFF',
                  weight='bold',
                  # style='italic',
                  fontsize=14)

    # 南北半球
    ax[0, 0].axhline(y=90, color='#000000')
    ax[0, 0].text(0.25, 122,
                  'North',
                  color='#FFFFFF',
                  weight='bold',
                  # style='italic',
                  fontsize=14,
                  rotation=90)
    ax[0, 0].text(0.25, 30,
                  'South',
                  color='#FFFFFF',
                  weight='bold',
                  # style='italic',
                  fontsize=14,
                  rotation=90)

    # カラーバー
    axpos = ax[0, 1].get_position()
    pp_ax = fig.add_axes(
        [0.84, axpos.y0, 0.02, axpos.height])  # カラーバーのaxesを追加
    pp = fig.colorbar(mappable0, cax=pp_ax)
    if MODE == 'FLUX':
        # pp.set_label('Precipitation Flux [cm$^{-2}$ s$^{-1}$]', fontsize=12)
        pp.set_label('Flux [cm-2 s-1]', fontsize=15)
    if MODE == '1356':
        pp.set_label('Brightness [Rayleigh]', fontsize=15)
    pp.ax.tick_params(labelsize=15)
    pp.ax.yaxis.get_offset_text().set_fontsize(15)
    pp.ax.yaxis.set_major_formatter(
        ptick.ScalarFormatter(useMathText=True))    # 指数表記
    pp.ax.yaxis.set_offset_position('left')
    # fig.tight_layout()

    if save_yn == 1:
        plt.savefig('P'+str(code)+'_'+str(date) +
                    '_alp_'+str(alp)+'_'+str(MODE)+'_2.png')

    plt.show()
    # plt.close()

    return 0


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
    # a0[:, 6] ... yn (=1)
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
    energy = a0[:, 7]
    aeq = a0[:, 8]
    vdotn = a0[:, 9]

    return xyz, energy, aeq, vdotn


# 積分する
if MODE == '1356':
    # EMISSION RATES
    dev = devlist
    fv2 = maxwell(enlist)
    v1 = np.sqrt((enlist+dev)*2*(1.6E-19)/me)
    v2 = np.sqrt((enlist)*2*(1.6E-19)/me)
    dv = v1 - v2
    fv2dv = fv2*dv  # 無次元 mからcmに変換してはいけない
    v2 *= 100       # 速さの次元 mからcmに変換しなければならない
    EM = np.sum(crosssections*(1E-18)*v2*fv2dv)
    # EM = np.sum(Qx(enlist)*v2*fv2dv)
    # print(Qx(enlist)*(1E+18))

# ピッチアングル分割数
pitch = 60
if ION_ELECTRON == 1:
    pitch *= 2

# 描画
# ヒストグラム初期化
H = 0
H3d = np.zeros((len(enlist), 40-1, 80-1))
gamma3d = np.zeros((len(enlist), 40-1, 80-1))
# ヒストグラム積分(v方向)
for i in range(len(enlist)):
    filepath0 = str(code)+'_'+str(date)+'_' + \
        str(enlist[i])+'ev_alp_'+str(alp)+'.txt'
    xyz0, energy0, aeq0, vdotn0 = dataload(filepath0)

    # 緯度経度ヒストグラム作成
    rxyza = Europacentric(xyz0, aeq0)   # Europa中心座標に
    H0, X, Y = find(rxyza, energy0, aeq0, vdotn0)

    dev = devlist[i]
    fv2 = maxwell(enlist[i])
    v1 = np.sqrt((enlist[i]+dev)*2*(1.6E-19)/me)
    v2 = np.sqrt((enlist[i])*2*(1.6E-19)/me)
    dv = v1 - v2
    fv2dv = fv2*dv          # 無次元 mからcmに変換してはいけない
    # f_dalpha = 1/pitch    # 各ビンから放出する粒子の数で決まる。数えるときの重みづけ
    print(str(enlist[i])+'eV fv =', fv2)

    H += H0*fv2dv           # vで積分 → 密度orフラックスの計算に使う

    H3d[i, :, :] = H0*fv2dv  # あるエネルギーを持つ粒子の割合(1)

    gamma3d[i, :, :] = (crosssections[i]*(1E-18))*(v2*100)  # まだ sigma*v

    # H += H0*fv2dv*f_dalpha

if MODE == 'FLUX':
    H *= ne  # 各ビンの電子フラックス[cm-2 s-1]

if MODE == '1356':
    H3d_sum = np.sum(H3d, axis=0)   # 各ビンで(1)の総和をとる(=ほぼ速度分布)
    H3d_coef = 1/H3d_sum            # 各ビンで速度分布の積分が1になるような係数

    H3d_new = H3d*H3d_coef          # あるエネルギーを持つ電子の分布 f*dv
    # H3d_sum = np.sum(H3d_new, axis=0)         # 確認用(onesになるはず)
    # print(np.min(H3d_sum), np.max(H3d_sum))   # 確認用(ちゃんとonesになる)

    gamma3d *= H3d_new              # 各ビンにおける、あるエネルギーを持つ電子のemission rate(2)
    gamma2d = np.sum(gamma3d, axis=0)   # (2)をv方向に積分
    print(np.min(gamma2d), np.average(gamma2d), np.max(gamma2d))

    H *= ne  # 各ビンの電子密度[cm-3]
    H *= (1E-6)*(1/3.1415)*gamma2d*nc_O2  # 発光強度[Rayleigh]
    # H *= (1E-6)*(1/3.1415)*EM*nc_O2  # 発光強度[Rayleigh]

    # 最大値と最小値
print(mode)
print('======')
print('Max: ', np.max(H))
print('min: ', np.min(H))
print('ave: ', np.average(H))

# 先行後行半球の比を計算
H_trailing = H[:, 0:int(H.shape[1]/2)]
H_leading = H[:, int(H.shape[1]/2):]
print('== HEMISPHERE ==')
print('Trailing ave.: ', np.average(H_trailing))
print('Leading ave.: ', np.average(H_leading))
print('Trailing/Leading: ', np.average(H_trailing)/np.average(H_leading))

# 南北半球の比を計算
H_northhemi = H[0:int(H.shape[0]/2), :]
H_southhemi = H[int(H.shape[0]/2):, :]
print('== HEMISPHERE ==')
print('South ave.: ', np.average(H_southhemi))
print('North ave.: ', np.average(H_northhemi))
print('South/North: ', np.average(H_southhemi)/np.average(H_northhemi))

# 南北極域30度を比較
H_north30 = H[0:8, :]
H_south30 = H[-8:, :]
print('== POLE ==')
print('South 30 ave.: ', np.average(H_south30))
print('North 30 ave.: ', np.average(H_north30))
print('South30/North30: ', np.average(H_south30)/np.average(H_north30))

# 後行半球の南北極域を比較
H_trail_north30 = H[0:int(H.shape[0]/2), 0:int(H.shape[1]/2)]
H_trail_south30 = H[int(H.shape[0]/2):, 0:int(H.shape[1]/2)]
print('== TRAILING ==')
print('South ave.: ', np.average(H_trail_south30))
print('North ave.: ', np.average(H_trail_north30))
print('Trailing south/north: ',
      np.average(H_trail_south30)/np.average(H_trail_north30))

# 後行半球の南北極域30度を比較
H_trail_north30 = H[0:8, 0:int(H.shape[1]/2)]
H_trail_south30 = H[-8:, 0:int(H.shape[1]/2)]
print('== TRAILING ==')
print('South 30 ave.: ', np.average(H_trail_south30))
print('North 30 ave.: ', np.average(H_trail_north30))
print('Trailing south30/north30: ',
      np.average(H_trail_south30)/np.average(H_trail_north30))

# 描画
mapplot2(H, X, Y)

print('======')
print('Goodbye.')
