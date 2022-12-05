""" gc_contour.py

Created on Fri Jan 21 9:42:00 2022
@author: Shin Satoh

Description:
This program is intended to calculate the angles between
the surface grids on Europa and the local B field lines.

"""

# %% ライブラリのインポート
from pickletools import bytes1
from numba import jit, f8
from numba.experimental import jitclass
import numpy as np
import math
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
import time
from multiprocessing import Pool

# FAVORITE COLORS (FAVOURITE COLOURS?)
color = ['#6667AB', '#0F4C81', '#5B6770', '#FF6F61', '#645394',
         '#84BD00', '#F6BE00', '#F7CAC9', '#16137E', '#45B8AC']


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet} \usepackage{sansmath} \sansmath \usepackage{siunitx} \sisetup{detect-all}'
#    \usepackage{helvet}     # helvetica font
#    \usepackage{sansmath}   # math-font matching  helvetica
#    \sansmath               # actually tell tex to use it!
#    \usepackage{siunitx}    # micro symbols
#    \sisetup{detect-all}    # force siunitx to use the fonts


#
#
# %% SETTINGS FOR THE NEXT EXECUTION
lam = 10                    # degrees
lam_r = math.radians(lam)   # radians


#
#
# %% CONSTANTS
RJ = float(7E+7)        # Jupiter半径   単位: m
mJ = float(1.90E+27)    # Jupiter質量   単位: kg
RE = float(1.56E+6)     # Europa半径    単位: m
mE = float(4.8E+22)     # Europa質量    単位: kg
L94 = 9.4*RJ            # 衛星と木星中心の距離 単位: km

c = float(3E+8)         # 真空中光速    単位: m/s
me = float(9.1E-31)     # 電子質量      単位: kg
e = float(-1.6E-19)     # 電子電荷      単位: C

G = float(6.67E-11)     # 万有引力定数  単位: m^3 kg^-1 s^-2

mu = float(1.26E-6)     # 真空中透磁率  単位: N A^-2 = kg m s^-2 A^-2
Mdip = float(1.6E+27)   # Jupiterのダイポールモーメント 単位: A m^2


#
#
# %% 途中計算でよく出てくる定数の比
A1 = e/me                        # 運動方程式内の定数
A2 = (mu*Mdip)/(4*np.pi)         # ダイポール磁場表式内の定数
# A1 = float(-1.7582E+11)        # 運動方程式内の定数
# A2 = 1.60432E+20               # ダイポール磁場表式内の定数
A3 = 4*3.141592*me/(mu*Mdip*e)   # ドリフト速度の係数


#
#
# 木星とtrace座標系原点の距離(x軸の定義)
R0 = L94*(np.cos(np.radians(lam)))**(-2)
R0x = R0
R0y = 0
R0z = 0
R0vec = np.array([R0x, R0y, R0z])

# Europaのtrace座標系における位置
eurx = L94*math.cos(math.radians(lam)) - R0x
eury = 0 - R0y
eurz = L94*math.sin(math.radians(lam)) - R0z


#
#
# %% 高速な内積計算
@jit(nopython=True, fastmath=True)
def vecdot(vec1, vec2):
    """
    DESCRIPTION IS HERE.
    """
    dot = vec1[0, :, :]*vec2[0, :, :] + vec1[1, :, :] * \
        vec2[1, :, :] + vec1[2, :, :]*vec2[2, :, :]

    return dot


#
#
# %% 磁場
def Bfield(Rvec):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル
    """
    # x, y, zは木星からの距離
    x = Rvec[0]
    y = Rvec[1]
    z = Rvec[2]

    # distance
    R2 = x**2 + y**2
    r_5 = np.sqrt(R2 + z**2)**(-5)

    # Magnetic field (dipole)
    Bvec = A2*r_5*np.array([3*z*x, 3*z*y, 2*z**2 - R2])

    # Europa相対座標
    x1 = x - R0vec[0] - eurx
    y1 = y - R0vec[1] - eury
    z1 = z - R0vec[2] - eurz

    # 先に共回転方向にx軸を向ける
    x1, y1 = y1, -x1

    # Europaとの距離
    r = np.sqrt(x1**2 + y1**2 + z1**2)

    # x軸まわりの回転
    X1 = x1
    Y1 = y1*math.cos(lam_r) - z1*math.sin(lam_r)
    Z1 = y1*math.sin(lam_r) + z1*math.cos(lam_r)

    # 背景ダイポール
    Bx0 = Bvec[0]
    By0 = Bvec[1]
    Bz0 = Bvec[2]
    Bx0, By0 = By0, -Bx0

    # x軸まわりの回転
    BX0 = Bx0
    BY0 = By0*math.cos(lam_r) - Bz0*math.sin(lam_r)
    BZ0 = By0*math.sin(lam_r) + Bz0*math.cos(lam_r)

    # 誘導磁場の各成分
    BX1 = (2*X1**2 - Y1**2 - Z1**2)*BX0 + 3*X1*Y1*BY0
    BY1 = 3*X1*Y1*BX0 + (2*Y1**2 - Z1**2 - X1**2)*BY0
    BZ1 = 3*Z1*X1*BX0 + 3*Y1*Z1*BY0

    # x軸まわりの回転
    Bx1 = BX1
    By1 = BY1*math.cos(-lam_r) - BZ1*math.sin(-lam_r)
    Bz1 = BY1*math.sin(-lam_r) + BZ1*math.cos(-lam_r)

    # 係数
    A3 = -(RE**3)/(2*r**5)
    # print('A3: ', A3)

    # y軸を共回転方向に
    Bx1, By1 = -By1, Bx1

    Bind = A3*np.array([Bx1, By1, Bz1])
    # print('Bind: ', Bind)

    Bvec += Bind

    return Bvec


#
#
# %%
def Babs(Rvec):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル
    """
    # x, y, zは木星からの距離
    Bvec = Bfield(Rvec)
    B = np.sqrt(Bvec[0, :, :]**2 + Bvec[1, :, :]**2 + Bvec[2, :, :]**2)

    return B


#
#
# %%
def ax0(H, X, Y):
    # x軸ラベル(標準的なwest longitude)
    xticklabels = ['360$^\\circ$W', '270$^\\circ$W',
                   '180$^\\circ$W', '90$^\\circ$W', '0$^\\circ$']
    yticklabels = ['90$^\\circ$N', '45$^\\circ$N',
                   '0$^\\circ$', '45$^\\circ$S', '90$^\\circ$S']

    # 図のタイトル
    title = 'Latitude-West Longitude, $\\lambda=$' + \
        str(lam) + '$^\\circ$'

    # 描画
    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.set_aspect(1)
    # ax.set_title(title, fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    # ax.set_xticks(np.linspace(-180, 180, 5))
    # ax.set_yticks(np.linspace(0, 180, 5))
    # ax.set_xticklabels(xticklabels, fontsize=12)
    # ax.set_yticklabels(yticklabels, fontsize=12)
    ax.invert_yaxis()
    cs = ax.contour(X, Y, H, levels=15, colors='#ffffff')
    # cs = ax.contour(X, Y, H, levels=15, colors='#000000')
    ax.clabel(cs)

    ax.tick_params('x', length=0, which='major')  # 目盛りを消す
    ax.tick_params('y', length=0, which='major')  # 目盛りを消す
    plt.setp(ax.get_xticklabels(), visible=False)  # ラベルを消す
    plt.setp(ax.get_yticklabels(), visible=False)  # ラベルを消す
    ax.axis("off")                                # 全部消す

    fig.tight_layout()
    fig.savefig('gc_contour_lam10_induced.png', transparent=True)
    plt.show()

    return 0


#
#
# %%
def main():
    # 表面緯度経度
    s_colat0 = np.radians(np.linspace(0, 180, 100))
    s_long0 = np.radians(np.linspace(-180, 180, 2*s_colat0.size))

    # メッシュに
    s_long, s_colat = np.meshgrid(s_long0, s_colat0)

    # 表面法線ベクトル
    nvec = np.array([
        np.sin(s_colat)*np.cos(s_long),
        np.sin(s_colat)*np.sin(s_long),
        np.cos(s_colat)
    ])
    print(nvec.shape)

    # 法線ベクトルの回転
    x_rot = nvec[0, :, :]*math.cos(math.radians(-lam)) + \
        nvec[2, :, :]*math.sin(math.radians(-lam))
    y_rot = nvec[1, :, :]
    z_rot = -nvec[0, :, :]*math.sin(math.radians(-lam)) + \
        nvec[2, :, :]*math.cos(math.radians(-lam))
    nvec[0, :, :] = x_rot
    nvec[1, :, :] = y_rot
    nvec[2, :, :] = z_rot
    print(nvec.shape)

    # Trace座標系に
    Rinitvec = RE*nvec
    Rinitvec[0, :, :] += eurx
    Rinitvec[1, :, :] += eury
    Rinitvec[2, :, :] += eurz

    # 単位磁場ベクトル
    R0vec_array = np.ones(nvec.shape)
    R0vec_array[0, :, :] = R0x*R0vec_array[0, :, :]
    R0vec_array[1, :, :] = R0y*R0vec_array[1, :, :]
    R0vec_array[2, :, :] = R0z*R0vec_array[2, :, :]
    print(R0vec_array.shape)

    B = Babs(Rinitvec + R0vec_array)
    print(B.shape)
    B_array = np.ones(nvec.shape)
    B_array[0, :, :] = B*B_array[0, :, :]
    B_array[1, :, :] = B*B_array[1, :, :]
    B_array[2, :, :] = B*B_array[2, :, :]
    bvec = Bfield(Rinitvec + R0vec_array)/B_array

    # 表面法線ベクトルと単位磁場ベクトルの内積
    d = vecdot(nvec, bvec)
    print(np.max(d))
    print(np.min(d))

    # 角度[degrees]に変換
    arg = np.degrees(np.arccos(d))
    print(np.max(arg))
    print(np.min(arg))

    # 描画用メッシュ
    yedges = np.linspace(0, 180, 100)
    xedges = np.linspace(-180, 180, 2*yedges.size)
    X, Y = np.meshgrid(xedges, yedges)

    ax0(arg, X, Y)

    return 0


#
#
# %%
if __name__ == '__main__':
    a = main()
