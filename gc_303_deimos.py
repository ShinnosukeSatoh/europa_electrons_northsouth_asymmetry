""" gc_303_deimos.py

Created on Wed Mar 16 20:49:00 2022
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

THIS PROGRAM IS HIGHLY OPTIMIZED FOR A CALCULATION OF BRIGHTNESS OF 135.6 NM OI
EMISSIONS.

Version
1.0.0 (Mar 16, 2022)
1.1.0 (Apr 14, 2022) Io compatible
1.2.0 (Apr 22, 2022) Ip velocity field added

"""


# %% ライブラリのインポート
from numba import jit
from numba import objmode
# from numba.experimental import jitclass
import numpy as np
import math
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
import time
from multiprocessing import Pool

# SMTP
import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate

# from getpass import getpass


# FAVORITE COLORS (FAVOURITE COLOURS?)
color = ['#6667AB', '#0F4C81', '#5B6770', '#FF6F61', '#645394',
         '#84BD00', '#F6BE00', '#F7CAC9', '#16137E', '#45B8AC']


#
#
# %% TOGGLE
MOON = 'EUROPA'                    # IO, EUROPA, GANYMEDE
FORWARD_BACKWARD = -1          # 1=FORWARD, -1=BACKWARD
GYRO = 1                       # 0=GUIDING CENTER, 1=GYRO MOTION
ION_ELECTRON = 0               # 0=ELECTRON, 1=ION
Z = 2                          # CHARGE (-1 for ELECTRON)
U = 32                         # ATOMIC MASS (32 for SULFUR)
CORES = 79                     # NUMBER OF CPU CORES TO USE (1 - 79)
ALTITUDE = 2                   # ALTITUDE OF STARTING POINTS [m]
ALFVEN_THETA = np.radians(10)  # ANGLE BETWEEN B0 VECTOR AND ALFVEN WING


#
#
# %% SETTINGS FOR THE NEXT EXECUTION
date = '20220420e_EUROPA_test7'
eV_array = np.array([
    2, 4, 6, 8, 10,
    12, 14, 16, 18, 20,
    25, 30, 40, 50, 60, 80, 100,
    200, 300, 400, 500, 600, 800, 1000,
    2500, 5000, 7500,
    10000
])    # [eV]
alp = 0.1
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
e = Z*float(1.6E-19)    # 電荷 単位: C

mu = 1.26E-6            # 真空中透磁率  単位: N A^-2 = kg m s^-2 A^-2
Mdip = 1.6E+27          # Jupiterのダイポールモーメント 単位: A m^2
omgJ = 1.75868E-4       # 木星の自転角速度 単位: rad/s
omgR = omgJ-omgE        # 木星の衛星に対する相対的な自転角速度 単位: rad/s
# omgR = omgR*alp         # 減速した共回転の角速度 単位: rad/s
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
A3 = 4*3.141592*me/(mu*Mdip*e)   # ドリフト速度の係数


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

# 遠方しきい値(z方向) 磁気緯度で設定
z_p_rad = math.radians(11.0)      # 北側
z_m_rad = -math.radians(2.0)       # 南側
# z_p = R0*math.cos(z_p_rad)**2 * math.sin(z_p_rad)
# z_m = R0*math.cos(z_m_rad)**2 * math.sin(z_m_rad)


#
#
# %% 初期位置エリア(z=0)での速度ベクトル (つまり磁気赤道面でのピッチ角)
# V0 = math.sqrt((energy/me)*2*float(1.602E-19))
V0_array = np.sqrt((eV_array/me)*2*float(1.602E-19))    # ndarray
pitch = int(45)  # 0-180度を何分割するか
loss = 0.1     # Loss cone
alphaeq0 = np.radians(np.linspace(loss, 180-loss, int(pitch+1)))
alpha_array = (alphaeq0[1:] + alphaeq0[:-1])/2  # the middle values
d_alpha = np.abs(alpha_array[1]-alpha_array[0])


#
#
# %% Europa表面の初期座標
ncolat = 30  # 分割数
nphi = 60    # 分割数
long_array = np.radians(np.linspace(0, 360, nphi+1))
colat_array = np.radians(np.linspace(0, 180, ncolat+1))
# 動径方向の中点
long_middle = (long_array[1:] + long_array[:-1])*0.5   # r 中点
colat_middle = (colat_array[1:] + colat_array[:-1])*0.5   # r 中点
meshlong, meshcolat = np.meshgrid(long_middle, colat_middle)
# meshlong: 経度
# meshcolat: 余緯度
# ビンの中でシフトさせる距離(r-phiJ平面)
d_long = 0.5*np.abs(meshlong[0, 1] - meshlong[0, 0])
d_colat = 0.5*np.abs(meshcolat[1, 0] - meshcolat[0, 0])


#
#
# %% 計算終了通知の設定
def MAIL(sendAddress, pwd, calctime):
    subject = 'Notification from Deimos'
    bodyText = 'A series of calculation is done. Calculation time was ' + \
        str(calctime)+' sec. -- Deimos'
    fromAddress = sendAddress
    toAddress = 'shinnosuke.satoh@pparc.gp.tohoku.ac.jp'

    # SMTPサーバに接続
    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    smtpobj.starttls()
    smtpobj.login(sendAddress, pwd)

    # メール作成
    msg = MIMEText(bodyText)
    msg['Subject'] = subject
    msg['From'] = fromAddress
    msg['To'] = toAddress
    msg['Date'] = formatdate()

    # 作成したメールを送信
    smtpobj.send_message(msg)
    smtpobj.close()

    return 0


#
#
# %% 初期座標をビンの中でシフトさせる
@jit('Tuple((f8,f8))(f8,f8)', nopython=True, fastmath=True)
def dshift(mcolatr, mlongr):
    """
    DESCRIPTION IS HERE.
    """
    # ビンの中心からのずれ量 shapeは(ny,nx)
    colatshift = d_colat*(2*np.random.rand() - 1)
    longshift = d_long*(2*np.random.rand() - 1)

    # ビンの中心からずらした新しい座標
    mcolatr += colatshift
    mlongr += longshift

    return mcolatr, mlongr


#
#
# %% ピッチ角をシフトさせる
@jit('f8(f8)', nopython=True, fastmath=True)
def ashift(a):
    """
    DESCRIPTION IS HERE.
    """
    # ビンの中心からのずれ量 shapeは(ny,nx)
    da = d_alpha*(2*np.random.rand() - 1)

    # ピッチ角の代表値からずらした新しいピッチ角
    a += da

    return a


#
#
# %% 高速な内積計算
@jit('f8(f8[:],f8[:])', nopython=True, fastmath=True)
def vecdot(vec1, vec2):
    """
    `vec1` ... <ndarray> 3D vector \\
    `vec2` ... <ndarray> 3D vector
    """
    dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]

    return dot


#
#
# %% 高速なベクトル三重積
@jit('f8[:](f8[:],f8[:],f8[:])', nopython=True, fastmath=True)
def vec3X(vec1, vec2, vec3):
    """
    `vec1` ... <ndarray> 3D vector \\
    `vec2` ... <ndarray> 3D vector \\
    `vec3` ... <ndarray> 3D vector
    """
    cross3 = np.array([
        vec1[1]*(vec2[0]*vec3[1]-vec2[1]*vec3[0]) -
        vec1[2]*(vec2[2]*vec3[0]-vec2[0]*vec3[2]),
        vec1[2]*(vec2[1]*vec3[2]-vec2[2]*vec3[1]) -
        vec1[0]*(vec2[0]*vec3[1]-vec2[1]*vec3[0]),
        vec1[0]*(vec2[2]*vec3[0]-vec2[0]*vec3[2]) -
        vec1[1]*(vec2[1]*vec3[2]-vec2[2]*vec3[1])
    ])

    return cross3


#
#
# %% 任意の軸回りのベクトル回転
@jit('f8[:](f8[:],f8)', nopython=True, fastmath=True)
def Corotation(Rvec, theta):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル \\
    `theta` ... 共回転の回転角 [RADIANS]
    """
    n1 = eomg[0]
    n2 = eomg[1]
    n3 = eomg[2]

    cos = math.cos(theta)
    sin = math.sin(theta)

    Rmatrix = np.array([
        [(n1**2)*(1-cos)+cos, n1*n2*(1-cos)-n3*sin, n1*n3*(1-cos)+n2*sin],
        [n1*n2*(1-cos)+n3*sin, (n2**2)*(1-cos)+cos, n2*n3*(1-cos)-n2*sin],
        [n1*n3*(1-cos)-n2*sin, n2*n3*(1-cos)+n1*sin, (n3**2)*(1-cos)+cos]
    ])

    Rvec_new = np.array([
        Rmatrix[0, 0]*Rvec[0] + Rmatrix[0, 1]*Rvec[1] + Rmatrix[0, 2]*Rvec[2],
        Rmatrix[1, 0]*Rvec[0] + Rmatrix[1, 1]*Rvec[1] + Rmatrix[1, 2]*Rvec[2],
        Rmatrix[2, 0]*Rvec[0] + Rmatrix[2, 1]*Rvec[1] + Rmatrix[2, 2]*Rvec[2],
    ])

    # Rvec_new[2] = Rvec[2]

    return Rvec_new


#
#
# %% 磁場
@jit('f8[:](f8[:])', nopython=True, fastmath=True)
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
    r_5 = math.sqrt(R2 + z**2)**(-5)

    # Magnetic field
    Bvec = A2*r_5*np.array([3*z*x, 3*z*y, 2*z**2 - R2])

    return Bvec


#
#
@jit('f8(f8[:])', nopython=True, fastmath=True)
def Babs(Rvec):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル
    """
    # x, y, zは木星からの距離
    Bvec = Bfield(Rvec)
    B = math.sqrt(Bvec[0]**2 + Bvec[1]**2 + Bvec[2]**2)

    return B


#
#
# %% 電場
E0 = (L94*omgE)*Babs(np.array([eurx, eury, eurz])+R0vec)
E1 = alp*E0


def Efield(RV):
    """
    `RV` ... <ndarray> trace座標系 位置ベクトル
    """
    # 衛星との相対座標
    x = RV[0] - eurx
    y = RV[1] - eury
    z = RV[2] - eurz

    # y軸まわりにlamだけ回転(Z軸は衛星の南北に平行)
    lam_r = math.radians(lam)
    X = x*math.cos(lam_r)+z*math.sin(lam_r)
    Y = y
    # Z = -x*math.sin(lam_r)+z*math.cos(lam_r)

    # 衛星からの距離の二乗(円筒)
    R2 = X**2 + Y**2

    # Inside
    RC = RE
    if R2 <= RC**2:
        EX = E1  # 符号あってる?
        EY = 0
        EZ = 0

    # Outside
    elif R2 > RC**2:
        EX = -E0 + (E0-E1)*(RC**2)*(X+Y)*(Y-X)/(R2**2)
        EY = -(E0-E1)*(RC**2)*(2*X*Y)/(R2**2)
        EZ = 0

    # y軸まわりに-lamだけ回転
    Ex = EX*math.cos(-lam_r)+EZ*math.sin(-lam_r)
    Ey = EY
    Ez = -EX*math.sin(-lam_r)+EZ*math.cos(-lam_r)

    return np.array([Ex, Ey, Ez])


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

    # print('mirror point: ', np.degrees(lamu), np.degrees(la))

    return la


#
#
# %% 共回転角速度
@jit('f8(f8[:])', nopython=True, fastmath=True)
def OMEGA(Rvec):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル
    """
    Hlam = math.radians(10)  # スケールハイト的な

    r = math.sqrt(Rvec[0]**2+Rvec[1]**2+Rvec[2]**2)
    maglat = math.acos(Rvec[2]/r)   # unit: radian
    # print('MAGLAT: ', math.degrees(maglat))

    lamr = math.radians(90-lam)  # 余緯度に変換
    dist = 1-(1-alp)*np.exp(-((maglat-lamr)/Hlam)**2)

    omg = omgR*dist
    omg = omgR*alp

    # print('LOCAL ALPHA: ', dist, math.degrees(maglat))

    return omg


#
#
# %% 共回転ドリフト速度
@jit('f8[:](f8[:])', nopython=True, fastmath=True)
def Vdvector(Rvec):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル
    """

    """
    Vdvec = omgR*np.array([
        eomg[1]*Rvec[2] - eomg[2]*Rvec[1],
        eomg[2]*Rvec[0] - eomg[0]*Rvec[2],
        eomg[0]*Rvec[1] - eomg[1]*Rvec[0]
    ])

    # Ip 1995 Velocity
    RV = Rvec - R0vec   # Europa中心座標系

    # 衛星との相対座標
    x = RV[0] - eurx
    y = RV[1] - eury
    z = RV[2] - eurz

    # y軸まわりにlamだけ回転(Z軸は衛星の南北に平行)
    lam_r = -math.radians(lam)
    X = x*math.cos(lam_r)+z*math.sin(lam_r)
    Y = y
    # Z = -x*math.sin(lam_r)+z*math.cos(lam_r)

    # 衛星からの距離の二乗(円筒)
    R2 = math.sqrt(X**2 + Y**2)

    # Inside
    RC = RE    # 衛星表面からALTITUDE[m]

    # Inside
    V0 = math.sqrt((Rvec[0]**2)+(Rvec[1]**2)+(Rvec[2]**2))*(omgJ-omgE)
    if R2 <= RC**2:
        VX = 0
        VY = V0*alp
        # VZ = 0
        # print('INSIDE')

        # y軸まわりに-lamだけ回転
        Vx = VX*math.cos(-lam_r)  # +VZ*math.sin(-lam_r)
        Vy = VY
        Vz = -VX*math.sin(-lam_r)  # +VZ*math.cos(-lam_r)
        Vdvec = np.array([Vx, Vy, Vz])

        # +yと-yで速度場が全然違うわけで
        # そこをつなぐのが難しい

    # Outside
    elif (R2 > RC**2) and (R2 <= (4*RC)**2):
        VY = V0 + (1-alp)*V0*((RC/R2)**2)*(1-2*((Y/R2)**2))
        VX = -2*(1-alp)*V0*((RC/R2)**2)*(X*Y)/(R2**2)
        # VZ = 0
        # print('OUTSIDE')

        # y軸まわりに-lamだけ回転
        Vx = VX*math.cos(-lam_r)  # +VZ*math.sin(-lam_r)
        Vy = VY
        Vz = -VX*math.sin(-lam_r)  # +VZ*math.cos(-lam_r)
        Vdvec = np.array([Vx, Vy, Vz])

    elif R2 > (4*RC)**2:
        Vdvec = omgR*np.array([
            eomg[1]*Rvec[2] - eomg[2]*Rvec[1],
            eomg[2]*Rvec[0] - eomg[0]*Rvec[2],
            eomg[0]*Rvec[1] - eomg[1]*Rvec[0]
        ])
    """

    # 減速なし
    if Rvec[1] < -(RE+10):
        Vdvec = omgR*np.array([
            eomg[1]*Rvec[2] - eomg[2]*Rvec[1],
            eomg[2]*Rvec[0] - eomg[0]*Rvec[2],
            eomg[0]*Rvec[1] - eomg[1]*Rvec[0]
        ])

    # 減速あり
    elif Rvec[1] >= -(RE+10):
        Vdvec = OMEGA(Rvec)*np.array([
            eomg[1]*Rvec[2] - eomg[2]*Rvec[1],
            eomg[2]*Rvec[0] - eomg[0]*Rvec[2],
            eomg[0]*Rvec[1] - eomg[1]*Rvec[0]
        ])

    return Vdvec


#
#
# %% 遠心力項
@jit('f8[:](f8[:])', nopython=True, fastmath=True)
def centrif(Rvec):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル \\
    三重積 omgvec x (omgvec x Rvec) の計算
    """
    omg = OMEGA(Rvec)
    cross3 = (omg**2)*np.array([
        eomg[1]*(eomg[0]*Rvec[1]-eomg[1]*Rvec[0]) -
        eomg[2]*(eomg[2]*Rvec[0]-eomg[0]*Rvec[2]),
        eomg[2]*(eomg[1]*Rvec[2]-eomg[2]*Rvec[1]) -
        eomg[0]*(eomg[0]*Rvec[1]-eomg[1]*Rvec[0]),
        eomg[0]*(eomg[2]*Rvec[0]-eomg[0]*Rvec[2]) -
        eomg[1]*(eomg[1]*Rvec[2]-eomg[2]*Rvec[1])
    ])

    return cross3


#
#
# %% 自転軸からの距離 rho
@jit('f8(f8[:])', nopython=True, fastmath=True)
def Rho(Rvec):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル
    """
    Rlen2 = Rvec[0]**2 + Rvec[1]**2 + Rvec[2]**2
    Rdot = eomg[0]*Rvec[0] + eomg[1]*Rvec[1] + eomg[2]*Rvec[2]
    rho = math.sqrt(Rlen2 - Rdot**2)    # 三平方の定理

    return rho


#
#
# %% Alfven wing tube
@jit('UniTuple(f8,3)(f8,f8,f8)', nopython=True, fastmath=True)
def Alfven_tube(x, y, z):
    """
    `x` ... x position of a particle \\
    `y` ... y position of a particle \\
    `z` ... z position of a particle
    """

    # tubeの中心
    tubex = eurx
    tubey = z*np.tan(ALFVEN_THETA)
    tubez = z

    return tubex, tubey, tubez


#
#
# %% シミュレーションボックスの外に出た粒子の復帰座標を計算
@jit('f8[:](f8[:],f8,f8,f8)', nopython=True, fastmath=True)
def comeback(RV2, req, lam0, K0):
    """
    `RV2` ... <ndarray> trace座標系 \\
    `req` ... ダイポールからの距離(@磁気赤道面) \\
    `lam0` ... スタートの磁気緯度 \\
    `K0` ... 保存量 \\
    上下とも、ボックスの外に出るときにはその場の磁場はダイポール磁場に等しい
    """
    # trace座標系から木星原点位置ベクトルに変換
    Rvec = RV2[0:3] + R0vec

    # 共回転角速度
    # omg = omgR
    omg = OMEGA(Rvec)

    # 磁場ベクトルの単位ベクトル
    B0 = Babs(Rvec)
    bvec = Bfield(Rvec)/B0

    # 共回転ドリフト速度
    Vdvec = Vdvector(Rvec)
    Vdpara = vecdot(bvec, Vdvec)

    # 速度ベクトルの磁場に垂直な成分
    vperp = math.sqrt(
        2*K0/me - (RV2[3]-Vdpara)**2 + (Rho(Rvec)*omg)**2)

    # 速度ベクトルの磁場に平行な成分
    vparallel = RV2[3]      # 粒子の速度 + ドリフト速度
    vpara_particle = vparallel - Vdpara  # 粒子の速度のみ

    # 速さ更新
    veq = math.sqrt(2*K0/me + (Rho(Rvec)*omgR)**2)

    # その場のピッチ角
    alphau = math.atan2(vperp, vparallel)

    # ミラーポイントの磁気緯度
    mirlam = mirrorpoint(lam0, alphau)
    # print(lam0*180/3.1415, mirlam*180/3.1415)

    # 積分の刻み
    dellam = 1E-3

    # 積分の長さ
    lamlen = int((mirlam-lam0)/dellam)

    # 積分の台形近似
    # tau0 = 磁気緯度 lam0 からミラーポイントまでの所要時間
    tau0, tau1, tau2 = 0, 0, 0  # initialize
    for _ in range(lamlen):
        tau1, tau2 = 0, 0
        tau1 = math.cos(lam0) * math.sqrt(1+3*math.sin(lam0)**2) * \
            math.sqrt(1-((math.cos(mirlam) / math.cos(lam0))**6) *
                      math.sqrt((1+3*math.sin(lam0)**2)/(1+3*math.sin(mirlam)**2)))**(-1)
        lam0 += dellam
        tau2 = math.cos(lam0) * math.sqrt(1+3*math.sin(lam0)**2) * \
            math.sqrt(1-((math.cos(mirlam) / math.cos(lam0))**6) *
                      math.sqrt((1+3*math.sin(lam0)**2)/(1+3*math.sin(mirlam)**2)))**(-1)

        tau0 += 0.5*(tau1+tau2)*dellam
    tau0 = (req/veq)*tau0

    # tau 近似
    # tau_N = 0.5*((L94)/veq)*(3.7-1.6*np.sin(alphau))
    # print(tau0*2, tau_N)

    # 共回転復帰座標
    tau = FORWARD_BACKWARD*2*tau0
    Rvec_new = Corotation(Rvec, omg*tau)    # 復帰座標ベクトル

    # 復帰座標における共回転速度ベクトル
    B1 = Babs(Rvec_new)
    bvec = Bfield(Rvec_new)/B1
    Vdvec = Vdvector(Rvec_new)
    Vdpara = vecdot(bvec, Vdvec)
    # Vdperp = math.sqrt(Vdvec[0]**2 + Vdvec[1]
    #                    ** 2 + Vdvec[2]**2 - Vdpara**2)

    # ダイポール磁場だからその場の磁場強度でピッチ角出して
    # 速度ベクトルを振り直したらどう?
    # sin2_alpha_new = (B1/B0)*math.sin(alphau)**2
    # vparticle = math.sqrt(2*K0/me) - \
    #     math.sqrt(Vdvec[0]**2 + Vdvec[1]**2 + Vdvec[2]**2)

    RV2_new = np.zeros(RV2.shape)           # initialize
    RV2_new[0:3] = Rvec_new - R0vec         # 新しいtrace座標系の位置ベクトル
    RV2_new[3] = - vpara_particle + Vdpara       # 磁力線に平行な速度成分 向き反転
    # 磁力線に平行な速度成分 向き反転(こっちが正しい?)
    # RV2_new[3] = - vparticle*math.sqrt(1-sin2_alpha_new) + Vdpara
    # box外に出たときののvparallelと中に戻ってきたときのvparallelは厳密には違う
    # だからここで発散するんだと思う

    return RV2_new


#
#
# %% ジャイロを真剣に解く運動方程式
@jit('f8[:](f8[:],f8)', nopython=True, fastmath=True)
def ode1(r3d, t):
    """
    `r3d` ... <ndarray> trace座標系 \\
    `t` ... <float> 時刻
    """

    # r3d はジャイロ運動の位置ベクトルと速度ベクトル
    # r3d[0] ... x of gyration
    # r3d[1] ... y
    # r3d[2] ... z
    # r3d[3] ... vx
    # r3d[4] ... vy
    # r3d[5] ... vz

    # 木星原点の位置ベクトルに変換
    R3D = r3d[0:3] + R0vec

    # Magnetic Field
    # B = Babs(R3D)       # 磁場強度
    Bvec = Bfield(R3D)  # 磁場ベクトル

    # 共回転速度ベクトル
    vd = Vdvector(R3D)

    # 運動方程式の各成分(ジャイロ+共回転ドリフト)
    x = r3d[3] + vd[0]
    y = r3d[4] + vd[1]
    z = r3d[5] + vd[2]

    vx = A1*(r3d[4]*Bvec[2] - r3d[5]*Bvec[1])
    vy = A1*(r3d[5]*Bvec[0] - r3d[3]*Bvec[2])
    vz = A1*(r3d[3]*Bvec[1] - r3d[4]*Bvec[0])

    eom = np.array([
        x,
        y,
        z,
        vx,
        vy,
        vz
    ])

    return eom


#
#
# %% 回転中心位置ベクトルRについての運動方程式(eq.8 on Northrop and Birmingham, 1982)
@jit('f8[:](f8[:],f8, f8)', nopython=True, fastmath=True)
def ode2(RV, t, K0):
    """
    `RV` ... <ndarray> trace座標系 \\
    `t` ... 時刻 \\
    `K0` ... 保存量
    """
    # RV.shape >>> (6,)
    # 座標系 = Europa中心の静止系
    # RV[0] ... x of Guiding Center
    # RV[1] ... y
    # RV[2] ... z
    # RV[3] ... v parallel

    # 木星原点の位置ベクトルに変換
    Rvec = RV[0:3] + R0vec

    # Magnetic Field
    B = Babs(Rvec)          # 磁場強度
    bvec = Bfield(Rvec)/B   # 磁力線方向の単位ベクトル

    # 磁力線に平行な速度ベクトル
    Vparavec = RV[3]*bvec

    # 磁場強度の磁力線に沿った微分
    dX = 100.
    dY = 100.
    dZ = 100.
    ds = 100.
    dBds = (Babs(Rvec+ds*bvec) - B)/ds  # 微分の平行成分

    # 共回転角速度
    # omg = omgR
    omg = OMEGA(Rvec)

    # 遠心力項
    omgRxomgRxR_s = vecdot(bvec, -centrif(Rvec))  # 遠心力項の平行成分

    # 共回転ドリフト速度
    Vdvec = Vdvector(Rvec)     # 共回転ドリフト速度ベクトル
    Vdnorm = Vdvec[0]**2+Vdvec[1]**2+Vdvec[2]**2  # ノルム
    Vdpara = vecdot(bvec, Vdvec)    # 平行成分

    Vdvec2 = Vdvector(Rvec+ds*bvec)  # 共回転ドリフト速度ベクトル
    Vdnorm2 = Vdvec2[0]**2 + Vdvec2[1]**2 + Vdvec2[2]**2    # ノルム

    dsdVd2 = 0.5*(Vdnorm2 - Vdnorm)/ds  # 遠心力項
    # print(dsdVd2, omgRxomgRxR_s)

    # 微分の平行成分
    dVdparads = vecdot(
        bvec,
        np.array([
            (vecdot(bvec,
                    Vdvector(Rvec+np.array([dX, 0., 0.]))) - Vdpara)/dX,
            (vecdot(bvec,
                    Vdvector(Rvec+np.array([0., dY, 0.]))) - Vdpara)/dY,
            (vecdot(bvec,
                    Vdvector(Rvec+np.array([0., 0., dZ]))) - Vdpara)/dZ
        ])
    )

    # 係数 mu
    mu = (K0-0.5*me*(RV[3]-Vdpara)**2 + 0.5*me*(Rho(Rvec)*omg)**2)/B

    # parallel速度の微分方程式
    # dVparadt = -(mu/me)*dBds + omgRxomgRxR_s + (RV[3]-Vdpara)*dVdparads
    dVparadt = -(mu/me)*dBds + dsdVd2 + (RV[3]-Vdpara)*dVdparads

    # 新しい座標
    RVnew = np.array([
        Vparavec[0]+Vdvec[0],
        Vparavec[1]+Vdvec[1],
        Vparavec[2]+Vdvec[2],
        dVparadt
    ], dtype=np.float64)

    return RVnew


#
#
# %% 4次ルンゲクッタ.. functionの定義
@jit('f8[:](f8[:],f8[:], f8)', nopython=True, fastmath=True)
def rk4(Rinitvec, V0vec, V0):
    """
    `Rinitvec` ... <ndarray> trace座標系 \\
    `V0vec` ... <ndarray> 速度ベクトル \\
    `V0` ... <float> 初速
    """

    # 木星原点の位置ベクトルに変換
    Rvec = Rinitvec[0:3] + R0vec

    # 磁場と平行な単位ベクトル
    B = Babs(Rvec)
    bvec = Bfield(Rvec)/B

    # 自転軸からの距離 rho (BACKTRACING)
    rho = Rho(Rvec)
    Vdvec = Vdvector(Rvec)
    Vdpara = vecdot(bvec, Vdvec)  # 平行成分

    # Gyro Period
    TC = 2*np.pi*me/(np.abs(e)*B)
    # print('TC [sec]: ', TC)

    # Gyro Radius
    # gyroradius = (me*V0)/(np.abs(e)*B)

    # V0vec を分解
    vparallel = vecdot(bvec, V0vec)
    vperp = math.sqrt(V0**2 - vparallel**2)

    # 保存量 K0 (運動エネルギー)
    K0 = 0.5*me*((vparallel-Vdpara)**2 - (rho*omgR)**2 + vperp**2)

    # 初期座標&初期速度ベクトルの配列(rk4に突っ込む)
    # RV[0] ... x座標
    # RV[1] ... y座標
    # RV[2] ... z座標
    # RV[3] ... v parallel
    RV = np.array([
        Rinitvec[0],
        Rinitvec[1],
        Rinitvec[2],
        vparallel
    ])

    # 時刻初期化
    t = 0

    # ダイポールからの距離(@磁気赤道面 近似)
    r = math.sqrt(Rvec[0]**2 + Rvec[1]**2 + Rvec[2]**2)
    # req = r/(math.cos(lam)**2)
    req = r/(math.cos(math.radians(lam))**2)    # こっちが正しい

    # トレース開始
    dt = FORWARD_BACKWARD*TC
    dt2 = dt*0.5

    # 座標配列
    # trace[0] ... 終点 x座標
    # trace[1] ... 終点 y座標
    # trace[2] ... 終点 z座標
    # trace[3] ... yn
    # trace[4] ... 終点 energy [eV]
    # trace[5] ... 終点 alpha_eq [RADIANS]
    trace = np.zeros(6)

    # 有効な粒子か
    yn = 1  # 1...有効

    # ルンゲクッタ
    for k in range(200000000000):
        F1 = ode2(RV, t, K0)
        F2 = ode2(RV+dt2*F1, t+dt2, K0)
        F3 = ode2(RV+dt2*F2, t+dt2, K0)
        F4 = ode2(RV+dt*F3, t+dt, K0)
        RV2 = RV + dt*(F1 + 2*F2 + 2*F3 + F4)/6

        # 木星原点の位置ベクトルに変換
        Rvec1 = RV[0:3] + R0vec
        Rvec2 = RV2[0:3] + R0vec

        # 磁気緯度
        r_jovi1 = math.sqrt(Rvec1[0]**2 + Rvec1[1]**2 + Rvec1[2]**2)
        r_jovi2 = math.sqrt(Rvec2[0]**2 + Rvec2[1]**2 + Rvec2[2]**2)
        maglat1 = math.asin(Rvec1[2]/r_jovi1)   # 磁気緯度 unit: radian
        maglat2 = math.asin(Rvec2[2]/r_jovi2)   # 磁気緯度 unit: radian

        # Europaに再衝突
        r_eur = math.sqrt(
            (RV2[0]-eurx)**2 +
            (RV2[1]-eury)**2 +
            (RV2[2]-eurz)**2
        )

        # Europaに再衝突
        if r_eur < RE:
            yn = 0
            # print('Collide')
            break

        # Gyro period
        TC = 2*np.pi*me/(np.abs(e)*Babs(Rvec2))

        # Europaの近く
        if r_eur < 1.04*RE:
            # 時間刻みの更新
            dt = FORWARD_BACKWARD*TC
        else:
            # 時間刻みの更新
            dt = FORWARD_BACKWARD*50*TC

        # 時刻更新
        t += FORWARD_BACKWARD*dt
        dt2 = 0.5*dt

        # 木星に衝突
        if r_jovi2 < RJ:
            yn = 0
            print('Loss')
            break

        # 北側しきい値
        if (maglat1 < z_p_rad) and (maglat2 > z_p_rad):
            # print('UPPER')
            RV2 = comeback(RV2, req, maglat2, K0)

        # 南側しきい値
        if (maglat1 > z_m_rad) and (maglat2 < z_m_rad):
            # print('LOWER')
            RV2 = comeback(RV2, req, -maglat2, K0)

        # 磁気赤道面到達(これをプラズマシート中心にしたいんだけど...!)
        if (((RV[2] >= 0) and (RV2[2] < 0))           # 通過方向: N → S
                or ((RV[2] <= 0) and (RV2[2] > 0))):  # 通過方向: S → N
            # 遠方まできたか
            y = RV2[1]
            if y > -(RE+10):  # まだ近い
                # 座標更新
                RV = RV2
                continue
            else:        # 遠方まできた
                # print('North to south')
                bvec = Bfield(Rvec2)/Babs(Rvec2)
                Vdvec = Vdvector(Rvec2)
                Vdpara = vecdot(bvec, Vdvec)
                Vdperp = math.sqrt(Vdvec[0]**2 + Vdvec[1]**2
                                   + Vdvec[2]**2 - Vdpara**2)
                vperp = math.sqrt(
                    2*K0/me - (RV[3]-Vdpara)**2 + (Rho(Rvec2)*omgR)**2) - Vdperp
                vparallel = RV[3] - Vdpara
                Vnorm = math.sqrt(vparallel**2 + vperp**2)
                alpha_end = math.atan2(vperp, -vparallel)   # RADIANS
                energy_end = me*0.5*(Vnorm**2)/float(1.602E-19)

                # K1 = 0.5*me*((vparallel)**2 -
                #             (Rho(Rvec)*omgR)**2 + (vperp+Vdperp)**2)
                # print('alpha_end [degrees]: ', math.degrees(alpha_end))
                # print('energy_end [eV]: ', energy_end)
                # print('K1/K0: ', K1/K0)

                # 座標配列
                # trace[0] ... 終点 x座標
                # trace[1] ... 終点 y座標
                # trace[2] ... 終点 z座標
                # trace[3] ... yn
                # trace[4] ... 終点 energy [eV]
                # trace[5] ... 終点 alpha_eq [RADIANS]
                trace[0:3] = RV[0:3]
                trace[3] = yn
                trace[4] = energy_end
                trace[5] = alpha_end

                break

        # 座標更新
        RV = RV2

        # if abs(t) > 5000:
        #    print('out')
        #     break

    return trace


#
#
# %% 4次ルンゲクッタ.. functionの定義
@jit('f8[:](f8[:],f8[:],f8)', nopython=True, fastmath=True)
def rk4_hybrid(Rinitvec, V0vec, V0):
    """
    `Rinitvec` ... <ndarray> trace座標系 \\
    `V0vec` ... <ndarray> 速度ベクトル \\
    `V0` ... <float> 初速
    """
    # print('V0 [km/s]: ', V0/1000)
    # 位置ベクトル&速度ベクトル(6列)
    r3d = np.array([
        Rinitvec[0],    # x座標
        Rinitvec[1],
        Rinitvec[2],
        V0vec[0],       # vx
        V0vec[1],
        V0vec[2]
    ])

    # 時刻初期化
    t = 0

    # 木星原点座標(Rinitvecはtrace座標系)
    Rvec = Rinitvec + R0vec

    # ダイポールからの距離(@磁気赤道面 近似)
    # r = math.sqrt(Rvec[0]**2 + Rvec[1]**2 + Rvec[2]**2)
    # req = r/(math.cos(math.radians(lam))**2)
    req = ((Rvec[0]**2 + Rvec[2]**2)**(3/2)) / (Rvec[0]**2)

    # Gyro Period
    B = Babs(Rvec)
    TC = 2*np.pi*me/(np.abs(e)*B)
    dt = FORWARD_BACKWARD*TC/(100)   # ジャイロを180分割するような時間刻み
    dt2 = dt*0.5

    # Gyro Radius
    gyroradius = (me*V0)/(np.abs(e)*B)

    # 有効な粒子か
    yn = 1

    # 座標配列
    # trace[0] ... 終点 x座標
    # trace[1] ... 終点 y座標
    # trace[2] ... 終点 z座標
    # trace[3] ... yn
    # trace[4] ... 終点 energy [eV]
    # trace[5] ... 終点 alpha_eq [RADIANS]
    trace = np.zeros(6)

    # ルンゲクッタ(ジャイロ)
    # print('GYRO STARTS')
    for k in range(6000000000000000000):
        f1 = ode1(r3d, t)
        f2 = ode1(r3d+dt2*f1, t+dt2)
        f3 = ode1(r3d+dt2*f2, t+dt2)
        f4 = ode1(r3d+dt*f3, t+dt)
        r3d2 = r3d + dt*(f1 + 2*f2 + 2*f3 + f4)/6

        # Europaからの距離
        r_eur = math.sqrt(
            (r3d2[0]-eurx)**2 +
            (r3d2[1]-eury)**2 +
            (r3d2[2]-eurz)**2
        )

        # Europaに再衝突(地表面)
        if r_eur <= RE:
            yn = 0
            # print('Collide 1')
            break

        # 座標と速度更新
        r3d = r3d2

        # 時刻更新
        # t += FORWARD_BACKWARD*dt
        # Europaの近傍を出た
        if r_eur > (RE+ALTITUDE+(100*gyroradius)):
            # print('k:', k)
            break

        # if abs(r3d2[2]-eurz) > 1.2*RE:
        #     break

    t = 0
    if yn == 1:  # 有効な粒子
        # 木星原点座標(r3dはtrace座標系)
        Rvec = r3d[0:3] + R0vec

        # 磁場と平行な単位ベクトル
        B = Babs(Rvec)
        bvec = Bfield(Rvec)/B

        # 自転軸からの距離 rho (BACKTRACING)
        # rho = Rho(Rvec)
        Vdvec = Vdvector(Rvec)
        Vdpara = vecdot(bvec, Vdvec)  # 平行成分

        # Gyro Period
        TC = 2*np.pi*me/(np.abs(e)*B)
        # print('TC [sec]: ', TC)

        # 時間刻み
        dt = FORWARD_BACKWARD*TC
        dt2 = dt*0.5

        # 速度ベクトル
        V1vec = r3d[3:6]
        V1_2 = V1vec[0]**2 + V1vec[1]**2 + V1vec[2]**2

        # V0vec を分解
        vparallel = vecdot(bvec, V1vec)         # 磁力線平行成分
        vperp = math.sqrt(V1_2 - vparallel**2)  # 磁力線垂直成分

        # 保存量 K0 (運動エネルギー)
        # K0 = 0.5*me*((vparallel-Vdpara)**2 - (rho*omgR)**2 + vperp**2)    # これは誤り
        K0 = 0.5*me*((vparallel)**2 + vperp**2)

        # 初期座標&初期速度ベクトルの配列(回転中心計算用)
        # trace座標系
        # RV[0] ... x座標
        # RV[1] ... y座標
        # RV[2] ... z座標
        # RV[3] ... v parallel
        RV = np.array([
            r3d[0], r3d[1], r3d[2], vparallel   # vparallelの符号?
        ])
        # if vparallel < 0:
        #     # 速度ベクトル向きの確認
        #     print(vecdot(Bfield(Rinitvec+R0vec)/Babs(Rinitvec+R0vec), V0vec))

        # ルンゲクッタ(回転中心)
        # print('GUIDING CENTER STARTS')
        for k in range(6000000000000000000):
            F1 = ode2(RV, t, K0)
            F2 = ode2(RV+dt2*F1, t+dt2, K0)
            F3 = ode2(RV+dt2*F2, t+dt2, K0)
            F4 = ode2(RV+dt*F3, t+dt, K0)
            RV2 = RV + dt*(F1 + 2*F2 + 2*F3 + F4)/6

            # trace座標系から木星原点の位置ベクトルに変換
            Rvec1 = RV[0:3] + R0vec
            Rvec2 = RV2[0:3] + R0vec

            # Europa中心からの距離
            r_eur = math.sqrt(
                (RV2[0]-eurx)**2 +
                (RV2[1]-eury)**2 +
                (RV2[2]-eurz)**2
            )

            # Europaに再衝突
            if r_eur < RE:
                yn = 0
                # print('Collide 2')
                break

            # Gyro period
            TC = 2*np.pi*me/(np.abs(e)*Babs(Rvec2))

            # 時間刻みの更新
            dt = FORWARD_BACKWARD*TC*80
            dt2 = 0.5*dt

            # 時刻更新
            t += FORWARD_BACKWARD*dt

            # 磁気緯度
            r_jovi1 = math.sqrt(Rvec1[0]**2 + Rvec1[1]**2 + Rvec1[2]**2)
            r_jovi2 = math.sqrt(Rvec2[0]**2 + Rvec2[1]**2 + Rvec2[2]**2)
            maglat1 = math.asin(Rvec1[2]/r_jovi1)   # 磁気緯度 unit: radian
            maglat2 = math.asin(Rvec2[2]/r_jovi2)   # 磁気緯度 unit: radian

            # 木星に衝突
            if r_jovi2 < RJ:
                yn = 0
                print('Loss')
                break

            # 北側しきい値
            if (maglat1 < z_p_rad) and (maglat2 > z_p_rad):
                # print('UPPER', OMEGA(Rvec2)/omgR)
                RV2 = comeback(RV2, req, maglat2, K0)

            # 南側しきい値
            if (maglat1 > z_m_rad) and (maglat2 < z_m_rad):
                # print('LOWER', OMEGA(Rvec2)/omgR)
                RV2 = comeback(RV2, req, -maglat2, K0)

            # 磁気赤道面到達(これをプラズマシート中心にしたいんだけど...!)
            if (((RV[2] >= 0) and (RV2[2] < 0))           # 通過方向: N → S
                    or ((RV[2] <= 0) and (RV2[2] > 0))):  # 通過方向: S → N
                # 遠方まできたか
                y = RV2[1]
                if y > -(RE+10):  # まだ近い
                    # 座標更新
                    RV = RV2
                    continue
                else:        # 遠方まできた
                    # print('North to south')
                    bvec = Bfield(Rvec2)/Babs(Rvec2)
                    Vdvec = Vdvector(Rvec2)
                    Vdpara = vecdot(bvec, Vdvec)
                    Vdperp = math.sqrt(Vdvec[0]**2 + Vdvec[1]**2
                                       + Vdvec[2]**2 - Vdpara**2)
                    vperp = math.sqrt(
                        2*K0/me - (RV[3]-Vdpara)**2 + (Rho(Rvec2)*omgR)**2) - Vdperp
                    vparallel = RV[3] - Vdpara
                    Vnorm = math.sqrt(vparallel**2 + vperp**2)
                    alpha_end = math.atan2(vperp, -vparallel)   # RADIANS
                    energy_end = me*0.5*(Vnorm**2)/float(1.602E-19)

                    # K1 = 0.5*me*((vparallel)**2 -
                    #              (Rho(Rvec2)*omgR)**2 + (vperp+Vdperp)**2)
                    # print('alpha_end [degrees]: ', math.degrees(alpha_end))
                    # print('energy_end [eV]: ', energy_end)
                    # print('K1/K0: ', K1/K0)

                    # 座標配列
                    # trace[0] ... 終点 x座標
                    # trace[1] ... 終点 y座標
                    # trace[2] ... 終点 z座標
                    # trace[3] ... yn
                    # trace[4] ... 終点 energy [eV]
                    # trace[5] ... 終点 alpha_eq [RADIANS]
                    trace[0:3] = RV[0:3]
                    trace[3] = yn
                    trace[4] = energy_end
                    trace[5] = alpha_end

                    break

            # 座標更新
            RV = RV2

            if abs(t) > 500000:
                print('out')
                break

    return trace


#
#
# %% 4次ルンゲクッタ.. classの定義
class RK4:
    def __init__(self, Rinitvec, V0vec, V0):
        p = rk4_hybrid(Rinitvec, V0vec, V0)
        self.positions = p


#
#
# %% トレースを行うfunction
@jit(nopython=True, fastmath=True)
def calc(mcolatr, mlongr, V0):
    """ 
    # time.time() はそのままじゃ使えない
    with objmode(start0='f8'):
        start0 = time.perf_counter()
    """

    # result[:, 0] ... 出発点 x座標
    # result[:, 1] ... 出発点 y座標
    # result[:, 2] ... 出発点 z座標
    # result[:, 3] ... 終点 x座標
    # result[:, 4] ... 終点 y座標
    # result[:, 5] ... 終点 z座標
    # result[:, 6] ... yn
    # result[:, 7] ... 終点 energy [eV]
    # result[:, 8] ... 終点 alpha_eq [RADIANS]
    # result[:, 9] ... 出発点 v_dot_n
    result = np.zeros((len(alpha_array), 10))   # initialize

    # LOOP INDEX INITIALIZED
    i = 0
    for _ in range(len(alpha_array)):
        # 初期座標をシフトさせる
        mcolat, mlong = dshift(mcolatr, mlongr)

        # 表面法線ベクトル
        nvec = np.array([
            math.sin(mcolat)*math.cos(mlong),
            math.sin(mcolat)*math.sin(mlong),
            math.cos(mcolat)
        ])

        # 法線ベクトルの回転
        nvec = np.array([
            nvec[0]*math.cos(math.radians(-lam))+nvec[2] *
            math.sin(math.radians(-lam)),
            nvec[1],
            -nvec[0]*math.sin(math.radians(-lam))+nvec[2] *
            math.cos(math.radians(-lam))
        ])

        # 初期座標ベクトル(1m 上空に設定) =================
        # 表面への再衝突判定はもちろん表面で行う
        Rinitvec = (RE+ALTITUDE)*nvec
        # print(np.sqrt(Rinitvec[0]**2 + Rinitvec[1]**2 + Rinitvec[2]**2)/RE)

        # Trace座標系に
        Rinitvec += np.array([eurx, eury, eurz])

        # ダイポール原点座標系
        Rvec = Rinitvec + R0vec

        # 速度ベクトル V0vec
        # alpha: 0.01 以上 179.9 未満 でランダムに与える
        # beta: 0 以上 360 未満 でランダムに与える
        alpha = (179.9 - 0.01)*np.pi*np.random.rand() + 0.01
        beta = 2*np.pi*np.random.rand()
        V0vec = V0*np.array([
            math.sin(alpha)*math.cos(beta),
            math.sin(alpha)*math.sin(beta),
            math.cos(alpha)
        ])

        # V0vecベクトルの回転
        V0vec = np.array([
            V0vec[0]*math.cos(math.radians(-lam))+V0vec[2] *
            math.sin(math.radians(-lam)),
            V0vec[1],
            -V0vec[0]*math.sin(math.radians(-lam))+V0vec[2] *
            math.cos(math.radians(-lam))
        ])

        # 磁場と平行な単位ベクトル
        # B = Babs(Rvec)
        # bvec = Bfield(Rvec)/B

        # 自転軸からの距離 rho (BACKTRACING)
        # rho = Rho(Rvec)
        Vdvec = Vdvector(Rvec)
        # Vdpara = vecdot(bvec, Vdvec)  # 平行成分

        # 速度ベクトルと表面がなす角
        vdotn = vecdot(nvec, V0vec + Vdvec)

        # result[:, 0] ... 出発点 x座標
        # result[:, 1] ... 出発点 y座標
        # result[:, 2] ... 出発点 z座標
        # result[:, 3] ... 終点 x座標
        # result[:, 4] ... 終点 y座標
        # result[:, 5] ... 終点 z座標
        # result[:, 6] ... yn
        # result[:, 7] ... 終点 energy [eV]
        # result[:, 8] ... 終点 alpha_eq [RADIANS]
        # result[:, 9] ... 出発点 v_dot_n
        result[i, 0:3] = Rinitvec

        if GYRO == 1:
            result[i, 3:9] = rk4_hybrid(
                Rinitvec, V0vec, V0)    # ジャイロ→回転中心の順に
        else:
            result[i, 3:9] = rk4(Rinitvec, V0vec, V0)   # 回転中心だけ

        result[i, 9] = vdotn
        i += 1

    """
    if np.random.rand() > 0.85:    # ときどき計算時間を表示する
        with objmode():
            print('A BIN DONE [sec]: ',  (time.perf_counter() - start0))
    """

    return result


#
#
# %% 時間設定
t = 0
dt = float(1E-5)  # 時間刻みはEuropaの近くまで来たらもっと細かくして、衝突判定の精度を上げよう
t_len = 500000
# t = np.arange(0, 60, dt)     # np.arange(0, 60, dt)
tsize = int(t_len/dt)


#
#
# %% main関数
def main():
    # sendAddress = input('EMAIL ADRESS: ')
    # pwd = getpass('PASSWORD: ')

    # 初期座標
    mcolatr = meshcolat.reshape(meshcolat.size)  # 1次元化
    mlongr = meshlong.reshape(meshlong.size)  # 1次元化

    total_s = time.time()

    for i in range(len(eV_array)):
        savename = 'gc303_'+date+'_'+str(eV_array[i])+'ev_alp_'+str(alp)+'.txt'

        vabs = math.sqrt((eV_array[i]/me)*2*float(1.602E-19))
        gyroradius = (me*vabs)/(np.abs(e)*Babs(np.array([L94, 0, 0])))

        # 情報表示
        print('V0 [km/s]: {:>2f}'.format(vabs/1000))
        print('gyro radius [m]: {:>2f}'.format(gyroradius))
        print('alpha: {:>7d}'.format(alpha_array.size))
        print('ncolat: {:>7d}'.format(ncolat))
        print('nphi: {:>7d}'.format(nphi))
        print('total: {:>7d}'.format(alpha_array.size*ncolat*nphi))
        print(savename)

        # 並列計算用 変数リスト(zip)
        # np.arrayは不可。ここが1次元なのでpoolの結果も1次元。
        args = list(zip(mcolatr, mlongr, V0_array[i]*np.ones(mcolatr.shape)))

        # 並列計算の実行
        start = time.time()
        with Pool(processes=CORES) as pool:
            result_list = list(pool.starmap(calc, args))
        stop = time.time()
        print('%.3f sec' % (stop - start))

        # 返り値(配列 *行8列)
        # 結果をreshape
        result = np.array(result_list)
        result = result.reshape([alpha_array.size*ncolat*nphi, 10])
        # print(result.shape)
        # print(trace)

        # result[:, 0] ... 出発点 x座標
        # result[:, 1] ... 出発点 y座標
        # result[:, 2] ... 出発点 z座標
        # result[:, 3] ... 終点 x座標
        # result[:, 4] ... 終点 y座標
        # result[:, 5] ... 終点 z座標
        # result[:, 6] ... yn
        # result[:, 7] ... 終点 energy [eV]
        # result[:, 8] ... 終点 alpha_eq [RADIANS]
        # result[:, 9] ... 出発点 v_dot_n [m/s]

        # Europaに衝突しない(yn=1)の粒子を取り出す
        yn1 = np.where(result[:, 6] == 1)  # 0でない行を見つける(検索は表面yn=6列目)
        mageq = result[yn1]  # 0でない行だけ取り出し
        # print(mageq)

        # SAVE(LOCAL)
        np.savetxt(
            str(savename), mageq
        )

        """
        # SAVE(iCloud)
        np.savetxt(
            '/Users/shin/Library/Mobile Documents/com~apple~CloudDocs/PPARC/' +
            str(savename), mageq
        )
        """

        # 情報表示
        # print('alpha: {:>7d}'.format(alpha_array.size))
        # print('ncolat: {:>7d}'.format(ncolat))
        # print('nphi: {:>7d}'.format(nphi))
        # print('total: {:>7d}'.format(alpha_array.size*ncolat*nphi))
        print('allowed: {:>7d}'.format(mageq.shape[0]))
        # print(savename)

        # メモリ消去
        del result_list, result, yn1, mageq

        # print(str(eV_array[i])+'eV is done.')
        print('===================================')

    print('%.3f sec' % (time.time() - total_s))
    # MAIL(sendAddress, pwd, round(time.time() - total_s))

    return 0


#
#
# %% EXECUTE
if __name__ == '__main__':
    a = main()
