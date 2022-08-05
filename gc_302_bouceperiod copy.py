""" gc_302_bounceperiod.py

Created on Mon Mar 14 10:48 2022
@author: Shin Satoh

Description:
This program is intended to numerically solve the equation (30)
on Northrop and Birmingham (1982).
First order in gyroradius terms are omitted.
The equation (30) is a second-order ordinary differential equation
for guiding center motion of a charged particle moving in a
magnetic field with a corotation electric field.

The equation (30) is based on the assumption of
    vd <<< vc
            (vd: a corotation drift velocity)
            (vc: a cyclotron velocity)
so that you can divide the total velocity into a parallel term and
a perpendicular term.

This program is fully functional in both a forward-in-time tracig
and a backward-in-time tracing.

The first abiadic invariant is NOT conserved throughout half the
bounce motion along the field line. The kinetic energy K0 defined
as

    K = (1/2)*m*(v// - vd//)^2 + (1/2)*m*(rho*omega)^2 + (1/2)*m*vperp^2

is the invariant for the motion with a centrifugal force.

Particles start from grid points defined by 40 x 80 (latitude x longitude).
Each point has 60 different particles with different pitch angles. The
pitch angle are given randomly.

Due to the moon-plasma interaction, the bulk velocity (the corotation) slows
down near Europa. The strength of the interaction is described as $\\alpha$;
$\\alpha$ = 0 corresponds to the maximum interaction, and $\\alpha$ = 1
corresponds to no interaction.

We are interested only in the incident electrons on a flux tube that crosses
Europa. This fact simplifies the model of the moon-plasma interaction.  In
the simulation box, the corotation is uniformally defined by $\\alpha$.

This program is intended to calculate trajectories of both ions and electrons.
The gyro radius of a heavy ion, such as S++, is NOT negligible when tracing
the trajectories. In this program, the gyro motion is fully solved near Europa
and the guiding center is traced out side of the designated area.

Version
1.0.0 (Mar 14 2022)

"""


# %% ライブラリのインポート
import matplotlib as mpl
from matplotlib import rcParams
from numba import jit
# from numba import objmode
# from numba.experimental import jitclass
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
import time
# from multiprocessing import Pool

# SMTP
# import smtplib
# from email.mime.text import MIMEText
# from email.utils import formatdate
# from getpass import getpass


# FAVORITE COLORS (FAVOURITE COLOURS?)
color = ['#6667AB', '#0F4C81', '#5B6770', '#FF6F61', '#645394',
         '#84BD00', '#F6BE00', '#F7CAC9', '#16137E', '#45B8AC']


richtext = input('rich text (y) or (n): ')
if richtext == 'y':
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet} \usepackage{sansmath} \sansmath \usepackage{siunitx} \sisetup{detect-all}'
    #    \usepackage{helvet}     # helvetica font
    #    \usepackage{sansmath}   # math-font matching  helvetica
    #    \sansmath               # actually tell tex to use it!
    #    \usepackage{siunitx}    # micro symbols
    #    \sisetup{detect-all}    # force siunitx to use the fonts


#
#
# %% TOGGLE
MOON = 'EUROPA'        # IO, EUROPA, GANYMEDE
FORWARD_BACKWARD = -1  # 1=FORWARD, -1=BACKWARD
GYRO = 1               # 0=GUIDING CENTER, 1=GYRO MOTION
ION_ELECTRON = 1       # 0=ELECTRON, 1=ION
Z = 2                  # CHARGE (-1 for ELECTRON)
U = 32                 # ATOMIC MASS (32 for SULFUR)
CORES = 18             # NUMBER OF CPU CORES TO USE
kBT1 = 20               # [EUROPA] CENTER TEMPERATURE [eV] 95%
kBT2 = 250              # [EUROPA] CENTER TEMPERATURE [eV] 5%


#
#
# %% SETTINGS FOR THE NEXT EXECUTION
date = '20220312e'
Eeq = np.array([
    1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
    6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14,
    15, 17.5, 20, 22.5, 25, 27.5, 30, 35, 40, 45, 50,
    55, 60, 65, 70, 75, 80, 85, 90, 95,
    100, 150, 200, 250, 300, 350,
    400, 450, 500, 600, 700,
    1000, 2000, 3000, 4000, 5000, 6000, 7000,
    10000, 20000, 30000, 40000, 50000, 70000,
    1E+5, 2E+5, 3E+5, 5E+5, 7E+5,
    1E+6, 2E+6, 3E+6, 5E+6, 7E+6,
    1E+7, 2E+7, 3E+7, 5E+7, 7E+7,
])                # unit: eV
alp = 0.025
lam = 10.0        # degrees


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

c = 3E+8                # 真空中光速    単位: m/s
G = 6.67E-11            # 万有引力定数  単位: m^3 kg^-1 s^-2

NA = 6.02E+23           # アボガドロ数
me = 9.1E-31            # 電子質量   単位: kg
if ION_ELECTRON == 1:
    me = me*1836*U      # 荷電粒子質量 単位: kg
    print(me)
e = Z*(1.6E-19)         # 電荷      単位: C

mu = 1.26E-6            # 真空中透磁率  単位: N A^-2 = kg m s^-2 A^-2
Mdip = 1.6E+27          # Jupiterのダイポールモーメント 単位: A m^2
omgJ = 1.75868E-4       # 木星の自転角速度 単位: rad/s
omgJ = 2*np.pi/(9.95*3600)  # 木星の自転角速度 単位: rad/s
omgR = omgJ-omgE        # 木星の衛星に対する相対的な自転角速度 単位: rad/s
omgR = omgR*alp         # 減速した共回転の角速度 単位: rad/s
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
# Europaの中心を通る磁力線の脚(磁気赤道面)
R0 = L94
R0x = R0
R0y = 0
R0z = 0
R0vec = np.array([R0x, R0y, R0z])

# 初期条件座標エリアの範囲(木星磁気圏動径方向 最大と最小 phiJ=0で決める)
r_ip = (L94+1.15*RE)*(math.cos(math.radians(lam)))**(-2)
r_im = (L94-1.15*RE)*(math.cos(math.radians(lam)))**(-2)

# Europaのtrace座標系における位置
eurx = L94*math.cos(math.radians(lam)) - R0x
eury = 0 - R0y
eurz = L94*math.sin(math.radians(lam)) - R0z

# 遠方しきい値(z方向) 磁気緯度で設定
z_p_rad = math.radians(12.0)      # 北側
z_p = R0*math.cos(z_p_rad)**2 * math.sin(z_p_rad)
z_m_rad = math.radians(2.0)      # 南側
z_m = -R0*math.cos(z_m_rad)**2 * math.sin(z_m_rad)


#
#
# %% マクスウェル速度分布関数
@jit(nopython=True, fastmath=True)
def maxwell(kT):
    # kT: 電子温度 [eV]
    kT = kT*1.602E-19  # 電子温度[J]
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
# %% Newton法でミラーポイントの磁気緯度を調べる(ダイポール磁場)
@jit('f8(f8, f8)', nopython=True, fastmath=True)
def mirrorpoint(lamu, alphau):
    """
    `lamu` ... その場の磁気緯度 \\
    `alphau` ... その場のピッチ角 [RADIANS]
    """
    xn = math.radians(1E-5)  # 反復の開始値

    K = (math.sin(alphau)**2)*(math.cos(lamu)**6) * \
        (1+3*math.sin(lamu)**2)**(-0.5)

    # ニュートン法の反復
    for _ in range(50):
        f = math.cos(xn)**6 - math.sqrt(1+3*math.sin(xn)**2)*K
        fdash = -6*(math.cos(xn)**5)*math.sin(xn) - 3*(math.sqrt(1+3*math.sin(xn)
                                                                 ** 2)**(-1))*math.sin(xn)*math.cos(xn)*K
        xn += - f/fdash

    # xnは 360度以上 の数字になりうるので, 磁気緯度として相応しい値に変換する
    la = xn % (2*np.pi)
    if (la > 0.5*np.pi) and (la <= np.pi):
        la = np.pi - la
    elif (la > np.pi) and (la <= 1.5*np.pi):
        la += -np.pi
    elif (la > 1.5*np.pi) and (la < 2*np.pi):
        la = 2*np.pi - la

    # print('mirror point: ', np.degrees(la))

    return la


#
#
# %% シミュレーションボックスの外に出た粒子の復帰座標を計算
@jit(nopython=True, fastmath=True)
def comeback(req, lam0, veq, alphaeq):
    """
    ARGUMENT(S) \\
    `req` ...  ダイポールからの距離(@磁気赤道面) \\
    `lam0` ... スタートの磁気緯度(ここでは磁気赤道面) \\
    `veq` ... 初速[m/s] \\
    `alphaeq` ... スタートのピッチ角 \\
    -------------------- \\
    RETURN(s) \\
    `tau0` ... バウンス周期(shapeはveqと同じ)
    """

    # その場のピッチ角
    alphau = alphaeq

    # ミラーポイントの磁気緯度
    mirlam = mirrorpoint(lam0, alphau)
    print('mirror point [deg]', np.degrees(mirlam))
    # print(lam0*180/3.1415, mirlam*180/3.1415)

    # 積分の刻み
    dellam = 1E-3

    # 積分の長さ
    lamlen = int((mirlam-lam0)/dellam)

    # 積分の台形近似
    # tau0 = 磁気緯度 lam0 からミラーポイントまでの所要時間
    tau0, tau1 = 0, 0  # initialize
    for _ in range(lamlen):
        tau1 = math.cos(lam0) * math.sqrt(1+3*math.sin(lam0)**2) * \
            math.sqrt(1-((math.cos(mirlam) / math.cos(lam0))**6) *
                      math.sqrt((1+3*math.sin(lam0)**2)/(1+3*math.sin(mirlam)**2)))**(-1)
        lam0 += dellam
        tau1 += math.cos(lam0) * math.sqrt(1+3*math.sin(lam0)**2) * \
            math.sqrt(1-((math.cos(mirlam) / math.cos(lam0))**6) *
                      math.sqrt((1+3*math.sin(lam0)**2)/(1+3*math.sin(mirlam)**2)))**(-1)

        tau0 += tau1
    tau0 = (req/veq)*0.5*tau0*dellam
    # print(tau0.shape)

    # 半周期分に直す
    tau0 *= 2

    return tau0


#
#
# %% バウンス周期描画
def ax_bounce(Eeq, tau0, tau1):
    fig, ax = plt.subplots(
        2, 1,
        dpi=160,
        figsize=(5, 7),
        gridspec_kw={'height_ratios': [1.5, 3]}
    )

    fontsize = 15

    # Maxwellian
    ax[0].set_ylabel('Velocity distribution', fontsize=fontsize)
    ax[0].set_xscale('log')
    ax[0].set_xlim([1, np.max(Eeq)])
    ax[0].tick_params(axis='y', labelsize=fontsize)
    ax[0].plot(Eeq, maxwell(Eeq), color='#333333')
    ax[0].tick_params('x', length=0, which='major')  # 目盛りを消す
    plt.setp(ax[0].get_xticklabels(), visible=False)  # ラベルを消す

    # ax[1].set_title('Pitch angle $45^\\circ$', fontsize=17)
    ax[1].set_xlabel('Energy [eV]', fontsize=fontsize)
    ax[1].set_ylabel('Convection [$R_E$]', fontsize=fontsize)
    ax[1].set_xscale('log')
    ax[1].set_xlim([1, np.max(Eeq)])
    ax[1].set_ylim([0, 3*RE/1000])
    ax[1].tick_params(axis='x', labelsize=fontsize)
    ax[1].set_yticks(np.linspace(0, 5*RE/1000, 6))
    ax[1].set_yticklabels(
        np.array([0, 1, 2, 3, 4, None]), fontsize=fontsize)   # y軸のticks
    ax[1].plot(Eeq, tau0*100, color='#ef476f',
               linewidth=2, label='$\\alpha=1$')
    ax[1].plot(Eeq, tau0*100*0.5, color='#ffd166',
               linewidth=2, label='$\\alpha=0.5$')
    ax[1].plot(Eeq, tau0*100*0.25, color='#06d6a0',
               linewidth=2, label='$\\alpha=0.25$')
    ax[1].plot(Eeq, tau0*100*0.1, color='#118ab2',
               linewidth=2, label='$\\alpha=0.1$')
    ax[1].plot(Eeq, tau0*100*0.05, color='#073b4c',
               linewidth=2, label='$\\alpha=0.05$')
    # ax[1].plot(Eeq, tau0*100*0.025, color=color[8],
    #            label='$\\alpha=0.025$')
    ax[1].axhspan(0, (RE/1000)*2, color="olive",
                  alpha=0.1)  # 0<y<1500を塗りつぶす
    ax[1].legend(fontsize=14)
    # ax[0].tick_params('x', length=0, which='major')  # 目盛りを消す
    # plt.setp(ax[0].get_xticklabels(), visible=False)  # ラベルを消す

    fig.tight_layout()

    plt.subplots_adjust(hspace=.0)

    plt.savefig('maxwell_bounceperiod.png')

    plt.show()

    return 0


#
#
# %% main関数
def main():
    start = time.time()

    # 粒子の初期座標
    Rinitvec = R0vec
    req = np.sqrt((Rinitvec[0])**2 + (Rinitvec[1])**2 + (Rinitvec[2])**2)

    # ピッチ角
    alphaeq = np.radians(45)    # unit: radians

    # 粒子のエネルギー@磁気赤道面
    Keq = Eeq*(1.602E-19)       # unit: J

    # 荷電粒子の質量 unit: kg
    m_e = 9.1E-31
    m_s2 = m_e*1836*U

    # 速度@磁気赤道面
    # veq = np.sqrt(2*Keq/me)     # unit: m/s

    # バウンス半周期
    tau_e = comeback(req, 0, np.sqrt(3*Keq/m_e), alphaeq)
    # tau_s2 = comeback(req, 0, np.sqrt(3*Keq/m_s2), alphaeq)
    # print(tau_e)

    # 描画
    ax_bounce(Eeq, tau_e, tau_e)

    # 計算時間出力
    print('%.3f sec' % (time.time() - start))

    """
    # MOP用にパラメータalphaの妥当性に関するプロットを作成する
    # 相互作用の強さ
    a = np.linspace(0.005, 1, 100)

    # 通過時間
    vcor = a*(omgJ-omgE)*L94  # corotation velocity [m/s] with factor alpha
    # print('corotation [km/s]', vcor/1000)
    T_pass = 2*RE/vcor      # time to pass the diameter of the moon

    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].set_title('time ratio')
    ax[0].plot(a, T_pass)
    # 電子のエネルギーでループを回す
    for kT in [1, 10, 100, 1000]:
        v_e = np.sqrt(3*kT*e/m_e)
        print('electron velocity [m/s]', v_e)
        print('parallel velocity [m/s]', v_e*np.cos(np.radians(45)))

        # バウンス周期
        tau_e = comeback(req, 0, v_e, np.radians(45))
        print('half bounce period [sec]', tau_e)
        print('half bounce period [hour]', tau_e/3600)

        # 通過時間とバウンス半周期の比
        gamma = T_pass/(tau_e)

        ax[1].plot(a, gamma, label=str(kT)+'eV')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
    ax[1].legend()
    fig.tight_layout()
    plt.show()
    """

    return 0


#
#
# %% EXECUTE
if __name__ == '__main__':
    a = main()
