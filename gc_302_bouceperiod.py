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
MOON = 'IO'        # IO, EUROPA, GANYMEDE
FORWARD_BACKWARD = -1  # 1=FORWARD, -1=BACKWARD
GYRO = 1               # 0=GUIDING CENTER, 1=GYRO MOTION
ION_ELECTRON = 1       # 0=ELECTRON, 1=ION
Z = 2                  # CHARGE (-1 for ELECTRON)
U = 32                 # ATOMIC MASS (32 for SULFUR)
CORES = 18             # NUMBER OF CPU CORES TO USE


#
#
# %% SETTINGS FOR THE NEXT EXECUTION
date = '20220312e'
Eeq = np.array([
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    15, 20, 25, 30, 40, 50, 60, 70, 80, 90,
    100, 200, 300,
    400, 500, 700,
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

c = float(3E+8)         # 真空中光速    単位: m/s
G = float(6.67E-11)     # 万有引力定数  単位: m^3 kg^-1 s^-2

NA = 6.02E+23           # アボガドロ数
me = float(9.1E-31)     # 電子質量   単位: kg
if ION_ELECTRON == 1:
    me = me*1836*U      # 荷電粒子質量 単位: kg
    print(me)
e = Z*float(1.6E-19)    # 電荷      単位: C

mu = 1.26E-6            # 真空中透磁率  単位: N A^-2 = kg m s^-2 A^-2
Mdip = 1.6E+27          # Jupiterのダイポールモーメント 単位: A m^2
omgJ = 1.75868E-4       # 木星の自転角速度 単位: rad/s
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
    print(tau0.shape)

    # 半周期分に直す
    tau0 *= 2

    return tau0


#
#
# %% バウンス周期描画
def ax_bounce(veq, tau0, tau1):
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].set_title('Pitch angle $45^\\circ$', fontsize=12)
    ax[0].set_xlabel('Energy [eV]', fontsize=12)
    ax[0].set_ylabel('Bouce Period $\\times 0.5$ [s]', fontsize=12)
    ax[0].set_xscale('log')
    ax[0].set_xlim([1, np.max(veq)])
    ax[0].set_ylim([0, np.max(tau0)])
    ax[0].plot(veq, tau0, color=color[1], label='e$^-$')

    ax[1].set_title('Pitch angle $45^\\circ$', fontsize=12)
    ax[1].set_xlabel('Energy [eV]', fontsize=12)
    ax[1].set_ylabel('Convection [km]', fontsize=12)
    ax[1].set_xscale('log')
    ax[1].set_xlim([1, np.max(veq)])
    ax[1].set_ylim([0, 1E+7])
    ax[1].plot(veq, tau0*100, color=color[3], label='$\\alpha=1$')
    ax[1].plot(veq, tau0*100*0.5, color=color[4], label='$\\alpha=0.5$')
    ax[1].plot(veq, tau0*100*0.25, color=color[5], label='$\\alpha=0.25$')
    ax[1].plot(veq, tau0*100*0.1, color=color[6], label='$\\alpha=0.1$')
    ax[1].plot(veq, tau0*100*0.05, color=color[7], label='$\\alpha=0.05$')
    ax[1].plot(veq, tau0*100*0.025, color=color[8], label='$\\alpha=0.025$')
    ax[1].axhspan(0, (RE/1000)*2, color="olive", alpha=0.2)  # 0<y<1500を塗りつぶす
    ax[1].legend()

    ax[2].set_title('Pitch angle $45^\\circ$', fontsize=12)
    ax[2].set_xlabel('Energy [eV]', fontsize=12)
    ax[2].set_ylabel('Convection [km]', fontsize=12)
    ax[2].set_xscale('log')
    ax[2].set_xlim([1, np.max(veq)])
    ax[2].set_ylim([0, 30000])
    ax[2].plot(veq, tau0*100, color=color[3], label='$\\alpha=1$')
    ax[2].plot(veq, tau0*100*0.5, color=color[4], label='$\\alpha=0.5$')
    ax[2].plot(veq, tau0*100*0.25, color=color[5], label='$\\alpha=0.25$')
    ax[2].plot(veq, tau0*100*0.1, color=color[6], label='$\\alpha=0.1$')
    ax[2].plot(veq, tau0*100*0.05, color=color[7], label='$\\alpha=0.05$')
    ax[2].plot(veq, tau0*100*0.025, color=color[8], label='$\\alpha=0.025$')
    ax[2].axhspan(0, (RE/1000)*2, color="olive", alpha=0.2)  # 0<y<1500を塗りつぶす
    ax[2].legend()
    # ax_r = ax.twinx()
    # ax_r.set_ylabel('Bouce Period $\\times 0.5$ [s]', fontsize=12)
    # ax_r.set_ylim([0, np.max(tau1)*0.98])
    # ax_r.plot(veq, tau1, color=color[3], label='S++')

    # added these three lines
    # ax[0].legend(loc='upper right')
    # ax_r.legend(loc='lower left')

    fig.tight_layout()

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

    # バウンス周期
    tau_e = comeback(req, 0, np.sqrt(2*Keq/m_e), alphaeq)
    tau_s2 = comeback(req, 0, np.sqrt(2*Keq/m_s2), alphaeq)

    # 描画
    ax_bounce(Eeq, tau_e, tau_e)

    # 計算時間出力
    print('%.3f sec' % (time.time() - start))

    return 0


#
#
# %% EXECUTE
if __name__ == '__main__':
    a = main()
