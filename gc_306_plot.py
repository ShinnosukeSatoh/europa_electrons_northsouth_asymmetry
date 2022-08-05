""" gc_306_plot.py

Created on Mon Aug 1 2022
@author: Shin Satoh

Description:
Previous programs for plotting (e.g. gc_303_plot.py) include
`np.histogram2d` function, which enables a fast calculation of
particle number flux [cm-2 s-1]. However, the method cannot
derive the differential flux [cm-2 s-1 str-1 eV-1].
This program is intended to calculate differential flux without
using np.histogram2d method.

Version:
1.0.0 (Aug 1, 2022) Induced dipole field

"""


# %% ライブラリのインポート
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
kBT2 = 250              # [EUROPA] CENTER TEMPERATURE [eV] 5%
# kBT1 = 5              # [IO] CENTER TEMPERATURE [eV] 95%
# kBT2 = 5              # [IO] CENTER TEMPERATURE [eV] 5%
ne = 160                # ELECTRON DENSITY 160 cm-3, S++ DENSITY 20 cm-3
# ne = 2500             # [IO] ELECTRON DENSITY [cm-3]
nc_O2 = 1.4E+15         # O2 COLUMN DENSITY [cm-2]
# nc_O2 = 1.0E+16       # [IO] O2 COLUMN DENSITY [cm-2]


#
#
# %% SETTINGS FOR THE NEXT EXECUTION
code = 'gc306'
date = '20220803e_9_'+str(MOON)

eV_array = np.array([
    2, 4, 6, 8, 10, 12,
    14, 16, 18, 20,                 # Cross sectionあり
    25, 30, 40, 50, 60, 80, 100,    # Cross sectionあり
    200, 300, 400, 500, 600,        # Cross sectionあり
    800, 1000,
    2500, 5000,
    7500, 10000,
    25000, 50000, 75000
])    # [eV]

alp = 0.5
lam = 10.0        # degrees
# 'FLUX'...precipitation flux, '1356'...brightness, 'DIFF'...differential flux
MODE = 'FLUX'
pitch = 45        # ピッチアングル分割数
long = int(60)    # 経度分割数
lat = int(long/2)  # 緯度分割数
save_yn = 1       # save? y=1, n=0

# 励起断面積(14eVから600eVまで)
crosssections = np.array([
    0, 0, 0, 0, 0, 0,
    0.0003, 1.68, 2.73, 3.37, 4.11, 4.18, 4.72, 5.74,
    6.45, 6.76, 6.40, 4.79, 3.79, 3.13, 2.62, 2.21,
    0, 0,
    0, 0,
    0, 0,
    0, 0, 0
])

crosssections = np.array([
    0, 0, 0, 0, 0, 0,
    0.0003, 1.68, 2.73, 3.37, 4.11, 4.18, 4.72, 5.74,
    6.45, 6.76, 6.40, 4.79, 3.79, 3.13, 2.62, 2.21,
    2, 2,
    2, 2,
    2, 2,
    2, 2, 2
])

crosssections *= 1E-18

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
    25000, 25000, 25000
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
def maxwell(kT):
    # kT: 電子温度 [eV]
    kT *= 1.602E-19  # 電子温度[J]
    v = np.sqrt(3*kT/me)

    # 中心温度 20eV
    kBT = kBT1  # [eV]
    kBT = kBT*(1.602E-19)  # [J]
    fv20 = 4*np.pi*(v**2) * ((me/(2*np.pi*kBT))**(3/2)) * \
        np.exp(-(me*v**2)/(2*kBT))

    # 中心温度 300eV
    kBT = kBT2  # [eV]
    kBT = kBT*(1.602E-19)  # [J]
    fv300 = 4*np.pi*(v**2) * ((me/(2*np.pi*kBT))**(3/2)) * \
        np.exp(-(me*v**2)/(2*kBT))

    fv = 0.95*fv20 + 0.05*fv300

    return fv


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
def mapplot2(H, X, Y, REGION):
    plt.rcParams['figure.subplot.bottom'] = 0.18
    plt.rcParams['figure.subplot.right'] = 0.82

    # x軸ラベル(標準的なwest longitude)
    if REGION == 'ALL':
        figsize = (8, 4)
        xticks = np.linspace(-180, 180, 5)
        xticklabels = ['360 W', '270 W',
                       '180 W', '90 W', '0']
    if REGION == 'TRAILING':
        figsize = (6, 4)
        xticks = np.linspace(-180, 0, 3)
        xticklabels = ['360 W', '270 W',
                       '180 W']
    if REGION == 'LEADING':
        figsize = (6, 4)
        xticks = np.linspace(0, 180, 3)
        xticklabels = ['180 W', '90 W', '0']

    # y軸ラベル(緯度)
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

    # 図のタイトル
    if MODE == 'ENERGY':
        title = 'Electron Energy Flux, $\\lambda=$' + \
            str(lam) + '$^\\circ$'

    # 描画
    fig, ax = plt.subplots(
        2, 2,
        dpi=150,
        figsize=figsize,
        gridspec_kw={'width_ratios': [1, 18], 'height_ratios': [9, 1]}
    )

    # 左上 [0,0]
    ax[0, 0].set_ylabel('Latitude [deg]', fontsize=17)  # y軸のタイトル
    ax[0, 0].set_yticks(np.linspace(int(np.min(Y)), int(np.max(Y)), 5))
    ax[0, 0].set_yticklabels(yticklabels, fontsize=17)   # y軸のticks
    ax[0, 0].set_facecolor('#555555')    # グレーで塗りつぶす

    # 右上 [0,1]
    ax[0, 1].set_title(title, fontsize=17, weight='bold')    # 図のタイトル
    ax[0, 1].invert_yaxis()
    if (MODE == 'FLUX'):
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

    if (MODE == 'ENERGY'):
        mappable0 = ax[0, 1].pcolormesh(X, Y, H,
                                        cmap='viridis',
                                        vmin=0
                                        )

    # 左下 [1,0]
    # ax[1, 0].set_facecolor('#FFFFFF')  # 黒で塗りつぶす

    # 右下 [1,1]
    ax[1, 1].set_xlabel('Longitude [deg]', fontsize=17)    # x軸のタイトル
    ax[1, 1].set_xticks(xticks)
    ax[1, 1].set_xticklabels(xticklabels, fontsize=17)   # x軸のticks
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
    ax[1, 1].text(-124, 0.25,
                  'Trailing',
                  color='#FFFFFF',
                  weight='bold',
                  # style='italic',
                  fontsize=17)
    ax[1, 1].text(63, 0.25,
                  'Leading',
                  color='#FFFFFF',
                  weight='bold',
                  # style='italic',
                  fontsize=17)

    # 南北半球
    ax[0, 0].axhline(y=90, color='#000000')
    ax[0, 0].text(0.18, 117,
                  'North',
                  color='#FFFFFF',
                  weight='bold',
                  # style='italic',
                  fontsize=17,
                  rotation=90)
    ax[0, 0].text(0.18, 28,
                  'South',
                  color='#FFFFFF',
                  weight='bold',
                  # style='italic',
                  fontsize=17,
                  rotation=90)

    # カラーバー
    axpos = ax[0, 1].get_position()
    pp_ax = fig.add_axes(
        [0.83, axpos.y0, 0.03, axpos.height])  # カラーバーのaxesを追加
    pp = fig.colorbar(mappable0, cax=pp_ax)
    if MODE == 'FLUX':
        # pp.set_label('Precipitation Flux [cm$^{-2}$ s$^{-1}$]', fontsize=12)
        pp.set_label('[cm-2 s-1]', fontsize=17)
    if MODE == '1356':
        pp.set_label('[Rayleigh]', fontsize=17)
    if MODE == 'ENERGY':
        pp.set_label('[eV cm-2 s-1]', fontsize=17)
    pp.ax.tick_params(labelsize=17)
    pp.ax.yaxis.get_offset_text().set_fontsize(17)
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
    vdotn = np.abs(a0[:, 9])

    return xyz, energy, aeq, vdotn, gridnum


# ヒストグラム初期化
H1d = np.zeros(long*lat)
H2d = np.zeros((lat, long))
gamma1d = np.zeros(long*lat)
gamma2d = np.zeros((lat, long))
gamma3d = np.zeros((len(enlist), lat, long))


# ヒストグラム積分(v方向)
for i in range(len(enlist)):
    # ヒストグラム初期化
    H2di = np.zeros((lat, long))
    gamma2di = np.zeros((lat, long))

    Ei = enlist[i]      # i番目のエネルギー [eV]
    dEi = devlist[i]    # i番目からi+1番目のエネルギー差 [eV]

    filepath0 = str(code)+'_'+str(date)+'_' + \
        str(Ei)+'ev_alp_'+str(alp)+'.txt'
    xyz0, energy0, aeq0, vdotn0, gridnum0 = dataload(filepath0)

    # 速度分布関数
    fv2 = maxwell(Ei)                       # [s/m]
    v1 = np.sqrt(3*(Ei+dEi)*(1.6E-19)/me)   # [m/s]
    v2 = np.sqrt(3*(Ei)*(1.6E-19)/me)       # [m/s]
    dv = v1 - v2                            # [m/s]

    # 速度ベクトルと法線ベクトルがなす角(theta_s)
    theta_s = np.arccos(vdotn0/v1)

    # 表面個数を数えあげ
    # j番目の格子点で数え上げる
    for j in range(long*lat):
        j_index = np.where(gridnum0 == j)   # j番目格子点のインデックス

        # 速度ベクトル関連
        vdotn1 = vdotn0[j_index]     # 速度ベクトルと法線ベクトルの内積 [m/s]
        vdotn1 *= 100                # 速度ベクトルと法線ベクトルの内積 [cm/s]
        vdotn1_sum = np.sum(vdotn1)  # 内積の和 [m/s]

        # ピッチ角関連
        fa2 = 1/np.pi       # ピッチ角分布
        # fa2 = np.sin(aeq0[j_index])/np.pi
        # fa2 = np.sin(aeq0[j_index])**0.5
        da = np.pi/pitch    # ピッチ角刻み
        db = 2*da

        # j番目格子点: エネルギー Ei~Ei+dEi [eV] の個数フラックス [cm-2 s-1]
        num_flux_j = np.sum(vdotn1*ne*fv2*fa2*dv*da)

        # j番目格子点: エネルギー Ei~Ei+dEi [eV] の電子数密度 [cm-3]
        density_j_dv = np.ones(vdotn1.shape)*ne*fv2*fa2*dv*da
        density_j = np.sum(density_j_dv)

        # j番目格子点: エネルギー Ei~Ei+dEi [eV] の発光強度 [R]
        brightness_j = 0
        brightness_j = np.sum(
            crosssections[i]*(v1*100)*density_j_dv)
        brightness_j *= nc_O2*(1E-6)
        # brightness_j *= (1E+6)*(1/(4*np.pi))

        # 立体角関連
        theta_si = theta_s[j_index]  # 速度ベクトルと法線ベクトルがなす角 [rad]
        # j番目格子点に入射することが可能な立体角 [rad]
        if theta_si.size > 2:  # NOT EMPTY
            # print(j, theta_si.size)
            omg_s = 2*np.pi*(-1)*(np.cos(np.max(theta_si)) -
                                  np.cos(np.min(theta_si)))

            omg_s = 2*np.pi*(-1)*(np.cos(np.max(theta_si)) - 1)
            # print(np.degrees(np.max(theta_si)))
            domg_s = np.sin(theta_si)*da*db  # ここ違うんよ
            omg_s = np.sum(domg_s)

            # j番目格子点: エネルギー Ei~Ei+dEi [eV] のdifferential flux [cm-2 s-1 str-1 eV-1]
            diff_j = np.sum(vdotn1*ne*fv2*fa2*dv*da/(domg_s*Ei))

        else:
            diff_j = 0

        if MODE == 'FLUX':
            H1d[j] = num_flux_j

        if MODE == '1356':
            H1d[j] = brightness_j

        if MODE == 'DIFF':
            H1d[j] = diff_j

    # 緯度経度マップにreshape
    H2di = H1d.reshape(lat, long)

    H2d += H2di

# H2dは横軸(経度方向)を正しく入れ替える必要あり
# 現状 ... 180W -> 90W -> 0W/360W -> 270W -> 180W
# これを ... 360W -> 270W -> 180W -> 90W -> 0W に直す
H2d = np.append(H2d[:, int(long/2):], H2d[:, 0:int(long/2)], axis=1)  # 変換した
# print(H2d)

trailing_hemi = H2d[:, 0:int(long/2)]
leading_hemi = H2d[:, int(long/2):]
print(str(filepath0))
print('MODE is', str(MODE))
print('=== HEMISPHERE ===')
print('Trailing   ', np.average(trailing_hemi))
print('Leading    ', np.average(leading_hemi))

# 30 degs polar regions on the leading hemisphere
north_30 = H2d[0:int(lat/6), int(long/2):]
south_30 = H2d[-int(lat/6):, int(long/2):]
ns_ratio = np.average(north_30) / np.average(south_30)
if ns_ratio < 1:
    ns_ratio = 1/ns_ratio
print('=== LEADING POLAR 30 DEG ===')
print('North ave. ', np.average(north_30))
print('South ave. ', np.average(south_30))
print('NS ratio   ', ns_ratio)

# 30 degs polar regions on the trailing hemisphere
north_30 = H2d[0:int(lat/6), 0:int(long/2)]
south_30 = H2d[-int(lat/6):, 0:int(long/2)]
ns_ratio = np.average(north_30) / np.average(south_30)
if ns_ratio < 1:
    ns_ratio = 1/ns_ratio
print('=== TRAILING POLAR 30 DEG ===')
print('North ave. ', np.average(north_30))
print('South ave. ', np.average(south_30))
print('NS ratio   ', ns_ratio)

# 描画 全球
x = np.linspace(0, 360, long+1)
y = np.linspace(0, 180, lat+1)
X, Y = np.meshgrid(x, y)
mapplot2(H2d, X, Y, 'ALL')

# 描画 Trailing
x = np.linspace(180, 360, trailing_hemi.shape[1]+1)
X, Y = np.meshgrid(x, y)
# mapplot2(trailing_hemi, X, Y, 'TRAILING')
