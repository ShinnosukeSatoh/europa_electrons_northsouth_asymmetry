""" gc_203_grid3.py

Created on Tue Feb 1 18:00:00 2022
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
and a backward-in-time tracing. WELL DONE!

Additional description on Jan 4 2022:
The first abiadic invariant is NOT conserved throughout half the
bounce motion along the field line. The kinetic energy K0 defined
as

    K = (1/2)*m*(v// - vd//)^2 + (1/2)*m*(rho*omega)^2 + (1/2)*m*vperp^2

is the invariant for the motion with a centrifugal force.

Particles start from grid points defined by 40 x 80 (latitude x longitude).
Each point has 60 different particles with different pitch angles. The
pitch angle are given randomly.

"""


# %% ライブラリのインポート
from numba import jit, f8, objmode
# from numba.experimental import jitclass
import numpy as np
import math
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
import time
from multiprocessing import Pool

# FAVORITE COLORS (FAVOURITE COLOURS?)
color = ['#6667AB', '#0F4C81', '#5B6770', '#FF6F61', '#645394',
         '#84BD00', '#F6BE00', '#F7CAC9', '#16137E', '#45B8AC']


#
#
# %% FORWARD OR BACKWARD
FORWARD_BACKWARD = -1  # 1=FORWARD, -1=BACKWARD


#
#
# %% SETTINGS FOR THE NEXT EXECUTION
energy = 10  # eV
savename = 'gc203g3d_'+str(energy)+'ev_alp_001_20220201.txt'
alp = 0.01


#
#
# %% CONSTANTS
RJ = float(7E+7)        # Jupiter半径   単位: m
mJ = float(1.90E+27)    # Jupiter質量   単位: kg
RE = float(1.56E+6)     # Europa半径    単位: m
mE = float(4.8E+22)     # Europa質量    単位: kg

c = float(3E+8)         # 真空中光速    単位: m/s
me = float(9.1E-31)     # 電子質量      単位: kg
e = float(-1.6E-19)     # 電子電荷      単位: C

G = float(6.67E-11)     # 万有引力定数  単位: m^3 kg^-1 s^-2

mu = float(1.26E-6)     # 真空中透磁率  単位: N A^-2 = kg m s^-2 A^-2
Mdip = float(1.6E+27)   # Jupiterのダイポールモーメント 単位: A m^2
omgJ = float(1.74E-4)   # 木星の自転角速度 単位: rad/s
omgE = float(2.05E-5)   # Europaの公転角速度 単位: rad/s
omgR = omgJ-omgE        # 木星のEuropaに対する相対的な自転角速度 単位: rad/s
eomg = np.array([-np.sin(np.radians(10.)),
                 0., np.cos(np.radians(10.))])
omgRvec = omgR*eomg
omgR2 = omgR
omgR2vec = omgR2*eomg


#
#
# %% 途中計算でよく出てくる定数の比
# A1 = float(e/me)                  # 運動方程式内の定数
# A2 = float(mu*Mdip/4/3.14)        # ダイポール磁場表式内の定数
A1 = float(-1.7582E+11)             # 運動方程式内の定数
A2 = 1.60432E+20                    # ダイポール磁場表式内の定数
A3 = 4*3.1415*me/(mu*Mdip*e)        # ドリフト速度の係数


#
#
# %% EUROPA POSITION (DETERMINED BY MAGNETIC LATITUDE)
lam = 10.0  # =============== !!! ==============
L96 = 9.6*RJ  # Europa公転軌道 L値

# 木星とtrace座標系原点の距離(x軸の定義)
# Europaの中心を通る磁力線の脚(磁気赤道面)
R0 = L96*(np.cos(np.radians(lam)))**(-2)
R0x = R0
R0y = 0
R0z = 0
R0vec = np.array([R0x, R0y, R0z])

# 初期条件座標エリアの範囲(木星磁気圏動径方向 最大と最小 phiJ=0で決める)
r_ip = (L96+1.15*RE)*(math.cos(math.radians(lam)))**(-2)
r_im = (L96-1.15*RE)*(math.cos(math.radians(lam)))**(-2)

# Europaのtrace座標系における位置
eurx = L96*math.cos(math.radians(lam)) - R0x
eury = 0 - R0y
eurz = L96*math.sin(math.radians(lam)) - R0z

# 遠方しきい値(z方向) 磁気緯度で設定
z_p_rad = math.radians(11.0)      # 北側
z_p = R0*math.cos(z_p_rad)**2 * math.sin(z_p_rad)
z_m_rad = math.radians(2.0)      # 南側
z_m = -R0*math.cos(z_m_rad)**2 * math.sin(z_m_rad)

# DEPLETION領域
depletionR = 1.5*RE  # 円筒の半径
mphi_leading = math.atan2(eury+R0y+1.05*RE, eurx+R0x)     # 先行半球中心の磁気経度
mphi_trailing = math.atan2(eury+R0y-1.05*RE, eurx+R0x)    # 後行半球中心の磁気経度
y_thresh = eury+R0y-1.1*RE


#
#
# %% 初期位置エリア(z=0)での速度ベクトル (つまり磁気赤道面でのピッチ角)
V0 = math.sqrt((energy/me)*2*float(1.602E-19))
pitch = int(60)  # 0-90度を何分割するか
alphaeq0 = np.radians(np.linspace(0.1, 179.9, int(pitch+1)))   # PITCH ANGLE
a0c = (alphaeq0[1:] + alphaeq0[:-1])/2  # the middle values
# alphaeq1 = np.radians(np.linspace(90.0, 179.9, int(pitch+1)))   # PITCH ANGLE
# a1c = (alphaeq1[1:] + alphaeq1[:-1])/2
alpha_array = a0c
d_alpha = np.abs(alpha_array[1]-alpha_array[0])


#
#
# %% Europa表面の初期座標
ncolat = 40  # 分割数   =========== !!!!! ===========
nphi = 80    # 分割数   =========== !!!!! ===========
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
# %% 初期座標をシフトさせる
@jit('Tuple((f8,f8))(f8,f8)', nopython=True, fastmath=True)
def dshift(mcolatr, mlongr):
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
    DESCRIPTION IS HERE.
    """
    dot = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]

    return dot


#
#
# %% 高速なベクトル三重積
@jit('f8[:](f8[:],f8[:],f8[:])', nopython=True, fastmath=True)
def vec3cross(vec1, vec2, vec3):
    """
    vec1 x (vec2 x vec3)
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
# %% Newton法でミラーポイントの磁気緯度を調べる(ダイポール磁場)
@jit('f8(f8, f8)', nopython=True, fastmath=True)
def mirrorpoint(lamu, alphau):
    """
    `lamu` ... その場の磁気緯度 \\
    `alphau` ... その場のピッチ角 [RADIANS]
    """
    xn = math.radians(1E-5)

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
# %% 自転軸からの距離 rho
@jit('f8(f8[:])', nopython=True, fastmath=True)
def Rho(Rvec):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル
    """
    Rlen2 = Rvec[0]**2 + Rvec[1]**2 + Rvec[2]**2
    Rdot = eomg[0]*Rvec[0] + eomg[1]*Rvec[1] + eomg[2]*Rvec[2]  # 自転軸との内積
    rho = math.sqrt(Rlen2 - Rdot**2)

    return rho


#
#
# %% 共回転ドリフト速度
@jit('f8[:](f8[:])', nopython=True, fastmath=True)
def Vdvector(Rvec):
    """
    DESCRIPTION IS HERE.
    """
    rv = Rvec - R0vec

    # Europa中心の座標系
    xyz = np.array([rv[0] - eurx, rv[1] - eury, rv[2] - eurz])

    Rc = RE
    # Vcor = omgR*Rho(Rvec)     # the corotating flow speed relative to Europa
    omgvec = omgR*eomg
    Vcor = omgvec[2]*Rvec[0] - omgvec[0]*Rvec[2]

    # 位置ベクトルの回転
    xyz = np.array([
        xyz[0]*math.cos(math.radians(-lam))+xyz[2] *
        math.sin(math.radians(-lam)),
        xyz[1],
        -xyz[0]*math.sin(math.radians(-lam))+xyz[2] *
        math.cos(math.radians(-lam))
    ])

    # Europa中心の座標系
    x = xyz[0]
    y = xyz[1]

    r = math.sqrt(x**2 + y**2)

    if r < Rc:
        Vx = 0
        Vy = alp * Vcor
        Vz = alp*(omgvec[0]*Rvec[1] - omgvec[1]*Rvec[0])

        V_ip = np.array([Vx, Vy, Vz])

        """
        # flow速度ベクトルの回転(trace座標系に戻す)
        V_ip = np.array([
            Vx*math.cos(math.radians(lam)),
            Vy,
            -Vx*math.sin(math.radians(lam))
        ])
        """

    else:
        # Vx = -2*(1-alp)*Vcor*(Rc**2)*(x*y)/(r**4)
        Vy = Vcor + (1-alp)*Vcor*((Rc/r)**2) * \
            ((x+y)*(x-y)/(r**2))   # 発散を避けるために変形
        # Vy = Vcor + (1-alp)*Vcor*((Rc/r)**2)*(1-2*(y**2)/(r**2))

        Vx = alp*(omgvec[1]*Rvec[2] - omgvec[2]*Rvec[1])
        Vz = alp*(omgvec[0]*Rvec[1] - omgvec[1]*Rvec[0])

        V_ip = np.array([Vx, Vy, Vz])

        # flow速度ベクトルの回転(trace座標系に戻す)
        """
        V_ip = np.array([
            Vx*math.cos(math.radians(lam)),
            Vy,
            -Vx*math.sin(math.radians(lam))
        ])
        """

    return V_ip


#
#
# %% 共回転ドリフト速度
@jit('f8(f8[:])', nopython=True, fastmath=True)
def Ip_Vx(Rvec):
    """
    DESCRIPTION IS HERE.
    """
    rv = Rvec - R0vec

    # Europa中心の座標系
    xyz = np.array([rv[0] - eurx, rv[1] - eury, rv[2] - eurz])

    Rc = RE
    # Vcor = omgR*Rho(Rvec)     # the corotating flow speed relative to Europa
    omgvec = omgR*eomg
    Vcor = omgvec[2]*Rvec[0] - omgvec[0]*Rvec[2]

    # 位置ベクトルの回転
    xyz = np.array([
        xyz[0]*math.cos(math.radians(-lam))+xyz[2] *
        math.sin(math.radians(-lam)),
        xyz[1],
        -xyz[0]*math.sin(math.radians(-lam))+xyz[2] *
        math.cos(math.radians(-lam))
    ])

    # Europa中心の座標系
    x = xyz[0]
    y = xyz[1]

    r = math.sqrt(x**2 + y**2)

    if r < Rc:
        Vx = 0

    else:
        Vx = -2*(1-alp)*Vcor*(Rc**2)*(x*y)/(r**4)

    return Vx


#
#
# %% Ip(1996)による減速効果モデル - 角速度を計算
@jit('f8(f8[:])', nopython=True, fastmath=True)
def Ip_omg(Rvec):
    """
    DESCRIPTION IS HERE.
    """
    rv = Rvec - R0vec

    # Europa中心の座標系
    xyz = np.array([rv[0] - eurx, rv[1] - eury, rv[2] - eurz])

    # 自転軸からの距離
    rho = Rho(Rvec)

    Rc = RE
    # Vcor = omgR*rho     # the corotating flow speed relative to Europa
    omgvec = omgR*eomg
    Vcor = omgvec[2]*Rvec[0] - omgvec[0]*Rvec[2]

    # flow速度ベクトルの回転
    xyz = np.array([
        xyz[0]*math.cos(math.radians(-lam))+xyz[2] *
        math.sin(math.radians(-lam)),
        xyz[1],
        -xyz[0]*math.sin(math.radians(-lam))+xyz[2] *
        math.cos(math.radians(-lam))
    ])

    # Europa中心の座標系
    x = xyz[0]
    y = xyz[1]

    r = math.sqrt(x**2 + y**2)

    if r < Rc:
        Vy = alp * Vcor

        # flow角速度
        omg_flow = Vy/rho     # 近似

    else:
        Vy = Vcor + (1-alp)*Vcor*((Rc/r)**2) * \
            ((x+y)*(x-y)/(r**2))   # 発散を避けるために変形
        # Vy = Vcor + (1-alp)*Vcor*((Rc/r)**2)*(1-2*(y**2)/(r**2))

        # flow角速度
        omg_flow = Vy/rho     # 近似

    return omg_flow


#
#
# %% 遠心力項
@jit('f8[:](f8,f8[:])', nopython=True, fastmath=True)
def centrif(omg, Rvec):
    """
    `omg` ... 角速度 [rad/s] \\
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル \\
    三重積 omgvec x (omgvec x Rvec) の計算
    """
    omgvec = omg*eomg
    cross3 = np.array([
        omgvec[1]*(omgvec[0]*Rvec[1]-omgvec[1]*Rvec[0]) -
        omgvec[2]*(omgvec[2]*Rvec[0]-omgvec[0]*Rvec[2]),
        omgvec[2]*(omgvec[1]*Rvec[2]-omgvec[2]*Rvec[1]) -
        omgvec[0]*(omgvec[0]*Rvec[1]-omgvec[1]*Rvec[0]),
        omgvec[0]*(omgvec[2]*Rvec[0]-omgvec[0]*Rvec[2]) -
        omgvec[1]*(omgvec[1]*Rvec[2]-omgvec[2]*Rvec[1])
    ])

    return cross3


#
#
# %% シミュレーションボックスの外に出た粒子の復帰座標を計算
@jit('f8[:](f8[:],f8,f8,f8)', nopython=True, fastmath=True)
def comeback(RV2, req, lam0, K0):
    """
    `RV2` ... <ndarray> trace座標系 \\
    `req` ... ダイポールからの距離(@磁気赤道面) \\
    `lam0` ... スタートの磁気緯度 \\
    `K0` ... 保存量
    """
    Rvec = RV2[0:3] + R0vec

    # flowの角速度
    omg = Ip_omg(Rvec)

    bvec = Bfield(Rvec)/Babs(Rvec)
    Vdvec = Vdvector(Rvec)
    Vdpara = vecdot(bvec, Vdvec)
    Vdperp = math.sqrt(Vdvec[0]**2 + Vdvec[1]
                       ** 2 + Vdvec[2]**2 - Vdpara**2)
    vperp = math.sqrt(
        2*K0/me - (RV2[3]-Vdpara)**2 + (Rho(Rvec)*omg)**2) - Vdperp
    vparallel = RV2[3] - Vdpara
    v_new = math.sqrt(vparallel**2 + vperp**2)
    veq = v_new

    # その場のピッチ角
    alphau = math.atan2(vperp, vparallel)

    # ミラーポイントの磁気緯度
    mirlam = mirrorpoint(lam0, alphau)

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

    # 共回転復帰座標
    tau = FORWARD_BACKWARD*2*tau0
    Rvec_new = Corotation(Rvec, omg*tau)

    # DEPLETION領域
    omg = Ip_omg(Rvec_new)

    # 保存量
    bvec = Bfield(Rvec_new)/Babs(Rvec_new)
    Vdvec = Vdvector(Rvec_new)
    Vdpara = vecdot(bvec, Vdvec)
    # Vdperp = math.sqrt(Vdvec[0]**2 + Vdvec[1]
    #                   ** 2 + Vdvec[2]**2 - Vdpara**2)
    # K1 = 0.5*me*((vparallel)**2 -
    #              (Rho(Rvec_new)*omgR)**2 + (vperp+Vdperp)**2)

    RV2_new = np.zeros(RV2.shape)
    RV2_new[0:3] = Rvec_new - R0vec         # 新しいtrace座標系の位置ベクトル
    RV2_new[3] = - vparallel + Vdpara       # 磁力線に平行な速度成分 向き反転
    RV2_new[4] = RV2[4]

    return RV2_new


#
#
# %% 回転中心位置ベクトルRについての運動方程式(eq.8 on Northrop and Birmingham, 1982)
@jit('f8[:](f8[:],f8, f8)', nopython=True, fastmath=True)
def ode2(RV, t, K0):
    """
    `RV2` ... <ndarray> trace座標系 \\
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

    omg = Ip_omg(Rvec)

    # 遠心力項
    omgRxomgRxR_s = vecdot(bvec, -centrif(omg, Rvec))  # 遠心力項の平行成分

    # 共回転ドリフト速度
    Vdvec = Vdvector(Rvec)     # 共回転ドリフト速度ベクトル
    Vdpara = vecdot(bvec, Vdvec)    # 平行成分

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
    dVparadt = -(mu/me)*dBds + omgRxomgRxR_s + (RV[3]-Vdpara)*dVdparads

    RVnew = np.array([
        Vparavec[0]+Ip_Vx(Rvec),  # =====================
        Vparavec[1]+Vdvec[1],
        Vparavec[2]+Vdvec[2],
        dVparadt
    ], dtype=np.float64
    )

    return RVnew


#
#
# %% 4次ルンゲクッタ.. functionの定義
@jit('f8[:](f8[:],f8, f8)', nopython=True, fastmath=True)
def rk4(RV0, tsize, TC):
    """
    `RV0` ... <ndarray> trace座標系 \\
    `tsize` ... 時刻tのサイズ \\
    `TC` ... サイクロトロン周期 [s] \\
    Details follow: \\
    `RV0.shape` ... (6,) \\
    `RV0[0]` ... x of Guiding Center \\
    `RV0[1]` ... y \\
    `RV0[2]` ... z \\
    `RV0[3]` ... v parallel \\
    `RV0[4]` ... K0 (保存量)
    """

    # 時刻初期化
    t = 0

    # K0 保存量
    K0 = RV0[4]

    # 木星原点の位置ベクトルに変換
    Rvec = RV0[0:3] + R0vec

    # ダイポールからの距離(@磁気赤道面 近似)
    r = math.sqrt(Rvec[0]**2 + Rvec[1]**2 + Rvec[2]**2)
    req = r/(math.cos(lam)**2)

    # トレース開始
    RV = RV0[0:4]
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
    # print('RK4 START')
    for k in range(2000000000000):
        F1 = ode2(RV, t, K0)
        F2 = ode2(RV+dt2*F1, t+dt2, K0)
        F3 = ode2(RV+dt2*F2, t+dt2, K0)
        F4 = ode2(RV+dt*F3, t+dt, K0)
        RV2 = RV + dt*(F1 + 2*F2 + 2*F3 + F4)/6

        # 木星原点の位置ベクトルに変換
        Rvec = RV2[0:3] + R0vec

        # Europaに再衝突
        r_eur = math.sqrt((RV2[0]-eurx)**2 + (RV2[1]-eury)
                          ** 2 + (RV2[2]-eurz)**2)
        if r_eur < RE:
            yn = 0
            # print('Collide')
            break

        # Gyro period
        TC = 2*np.pi*me/(-e*Babs(Rvec))
        # Europaの近く
        if r_eur < 1.03*RE:
            # 時間刻みの更新
            dt = FORWARD_BACKWARD*TC
        else:
            # 時間刻みの更新
            dt = FORWARD_BACKWARD*50*TC

        # 時間刻みの更新
        dt2 = 0.5*dt

        # 時刻更新
        t += dt

        # 木星に衝突
        # r_jovi = math.sqrt(Rvec[0]**2 + Rvec[1]**2 + Rvec[2]**2)
        # if r_jovi < RJ:
        #     yn = 0
        #     # print('Loss')
        #     break

        # 北側しきい値
        if (RV[2] < z_p) and (RV2[2] > z_p):
            # print('UPPER')
            RV2 = comeback(RV2, req, z_p_rad, K0)

        # 南側しきい値
        if (RV[2] > z_m) and (RV2[2] < z_m):
            # print('LOWER')
            RV2 = comeback(RV2, req, z_m_rad, K0)

        # 磁気赤道面到達 (N → S)
        if (RV[2] > 0) and (RV2[2] < 0):
            y = RV[1]
            if y > y_thresh:    # IN
                # 座標更新
                RV = RV2
                continue
            else:   # OUT OF
                # print('North to south')
                omg = Ip_omg(Rvec)

                bvec = Bfield(Rvec)/Babs(Rvec)
                Vdvec = Vdvector(Rvec)
                Vdpara = vecdot(bvec, Vdvec)
                Vdperp = math.sqrt(Vdvec[0]**2 + Vdvec[1]**2
                                   + Vdvec[2]**2 - Vdpara**2)
                vperp = math.sqrt(
                    2*K0/me - (RV[3]-Vdpara)**2 + (Rho(Rvec)*omg)**2) - Vdperp
                vparallel = RV[3] - Vdpara
                # print('vparallel 1: ', vparallel)
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

        # 磁気赤道面到達 (S → N)
        if (RV[2] < 0) and (RV2[2] > 0):
            y = RV[1]
            if y > y_thresh:    # IN
                # 座標更新
                RV = RV2
                continue
            else:   # OUT OF
                # print('South to north')
                omg = Ip_omg(Rvec)

                bvec = Bfield(Rvec)/Babs(Rvec)
                Vdvec = Vdvector(Rvec)
                Vdpara = vecdot(bvec, Vdvec)
                Vdperp = math.sqrt(Vdvec[0]**2 + Vdvec[1]**2
                                   + Vdvec[2]**2 - Vdpara**2)
                vperp = math.sqrt(
                    2*K0/me - (RV[3]-Vdpara)**2 + (Rho(Rvec)*omg)**2) - Vdperp
                vparallel = RV[3] - Vdpara
                # print('vparallel 2: ', vparallel)
                Vnorm = math.sqrt(vparallel**2 + vperp**2)
                alpha_end = math.atan2(vperp, -vparallel)   # RADIANS
                energy_end = me*0.5*(Vnorm**2)/float(1.602E-19)

                # K1 = 0.5*me*((vparallel)**2 -
                #              (Rho(Rvec)*omgR)**2 + (vperp+Vdperp)**2)
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

        if abs(t) > 5000:
            trace[0:3] = RV2[0:3]
            trace[3] = 0    # yn
            print('BREAK')
            break

    return trace


#
#
# %% 4次ルンゲクッタ.. classの定義
class RK4:
    def __init__(self, RV0, tsize, TC):
        mageq_positions = rk4(RV0, tsize, TC)
        self.positions = mageq_positions


#
#
# %% トレースを行うfunction
@jit(nopython=True, fastmath=True)
def calc(mcolatr, mlongr):
    # time.time() はそのままじゃ使えない
    with objmode(start0='f8'):
        start0 = time.perf_counter()

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
    result = np.zeros((len(alpha_array), 10))

    # LOOP INDEX INITIALIZED
    i = 0
    for alphar in alpha_array:
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

        # 表面ベクトル(Europa表面から10km上空にしてみる)
        Rinitvec = (RE)*nvec

        # Trace座標系に
        Rinitvec = Rinitvec + np.array([eurx, eury, eurz])

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
        B = Babs(Rinitvec + R0vec)
        bvec = Bfield(Rinitvec + R0vec)/B

        # 自転軸からの距離 rho (BACKTRACING)
        rho = Rho(Rinitvec + R0vec)
        Vdvec = Vdvector(Rinitvec + R0vec)
        Vdpara = bvec[0]*Vdvec[0] + bvec[1]*Vdvec[1] + bvec[2]*Vdvec[2]  # 平行成分

        # 速度ベクトルと表面がなす角
        vdotn = vecdot(nvec, V0vec + Vdvec)

        # Gyro Period
        TC = 2*np.pi*me/(-e*B)
        # print('TC [sec]: ', TC)

        vparallel = vecdot(bvec, V0vec)
        vperp = math.sqrt(V0**2 - vparallel**2)

        # 保存量 K0 (運動エネルギー)
        K0 = 0.5*me*((vparallel-Vdpara)**2 - (rho*omgR2)**2 + vperp**2)

        # 初期座標&初期速度ベクトルの配列
        # RV0vec[0] ... x座標
        # RV0vec[1] ... y座標
        # RV0vec[2] ... z座標
        # RV0vec[3] ... v parallel
        # RV0vec[4] ... K0 (保存量)
        RV0vec = np.array([
            Rinitvec[0], Rinitvec[1], Rinitvec[2], vparallel, K0
        ])

        # vdotn の判定を導入した新しい方式
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
        result[i, 3:9] += 10

        if vdotn < 0:
            # vdotn < 0 のときだけトレースを行う
            result[i, 3:9] = rk4(RV0vec, tsize, TC)
        result[i, 9] = vdotn
        i += 1

        """ # TRACING(古い方式)
        runge = rk4(RV0vec, tsize, TC)

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
        result[i, 3:9] = runge
        result[i, 9] = vdotn
        i += 1
        """
    # yn1 = np.where(result[:, 6] == 1)  # 0でない行を見つける(検索は表面yn=6列目)
    # result2 = result[yn1]
    # print('A BIN SHAPE: ', result2.shape)

    with objmode():
        print('A BIN DONE [sec]: ',  (time.perf_counter() - start0))

    return result


#
#
# %% 時間設定
t = 0
dt = float(1E-5)  # 時間刻みはEuropaの近くまで来たらもっと細かくして、衝突判定の精度を上げよう
t_len = 10000000
# t = np.arange(0, 60, dt)     # np.arange(0, 60, dt)
tsize = int(t_len/dt)


#
#
# %% main関数
def main():
    # 情報表示
    print('alpha: {:>7d}'.format(alpha_array.size))
    print('ncolat: {:>7d}'.format(ncolat))
    print('nphi: {:>7d}'.format(nphi))
    print('total: {:>7d}'.format(alpha_array.size*ncolat*nphi))
    print(savename)

    # 初期座標
    mcolatr = meshcolat.reshape(meshcolat.size)  # 1次元化
    mlongr = meshlong.reshape(meshlong.size)  # 1次元化

    # 並列計算用 変数リスト(zip)
    args = list(zip(mcolatr, mlongr))  # np.arrayは不可。ここが1次元なのでpoolの結果も1次元。

    # 並列計算の実行
    start = time.time()
    with Pool(processes=8) as pool:
        result_list = list(pool.starmap(calc, args))
    stop = time.time()
    print('%.3f seconds' % (stop - start))

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
        '/Users/shin/Documents/Research/Europa/Codes/gyrocenter/gyrocenter_2/' +
        str(savename), mageq
    )

    # SAVE(iCloud)
    np.savetxt(
        '/Users/shin/Library/Mobile Documents/com~apple~CloudDocs/PPARC/' +
        str(savename), mageq
    )

    # 情報表示
    print('alpha: {:>7d}'.format(alpha_array.size))
    print('ncolat: {:>7d}'.format(ncolat))
    print('nphi: {:>7d}'.format(nphi))
    print('total: {:>7d}'.format(alpha_array.size*ncolat*nphi))
    # print('magnetic equator: {:>7d}'.format(mageq.shape))
    print('#NaN: ', mageq[np.where(np.isnan(mageq).any(axis=1))].shape)
    print(savename)

    return 0


#
#
# %% EXECUTE
if __name__ == '__main__':
    a = main()
