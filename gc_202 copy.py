""" gc_202.py

Created on Sun Dec 26 11:40:00 2021
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

This program is fully functional in both a forward-in-time tracing
and a backward-in-time tracing. WELL DONE!

"""


# %% ライブラリのインポート
from numba import jit, f8
from numba.experimental import jitclass
import numpy as np
import math
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
import time
# from multiprocessing import Pool

# FAVORITE COLORS (FAVOURITE COLOURS?)
color = ['#6667AB', '#0F4C81', '#5B6770', '#FF6F61', '#645394',
         '#84BD00', '#F6BE00', '#F7CAC9', '#16137E', '#45B8AC']


#
#
# %% FORWARD OR BACKWARD
FORWARD_BACKWARD = -1  # 1=FORWARD, -1=BACKWARD


#
#
# %% 座標保存の間隔(hステップに1回保存する)
h = int(300)


#
#
# %% SETTINGS FOR THE NEXT EXECUTION
energy = 1000  # eV
savename_f = 'go_1000ev_aeq60_20211230_1_forward.txt'
savename_b = 'go_1000ev_aeq60_20211230_1_backward.txt'
alphaeq = np.radians(170)   # PITCH ANGLE


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
omgR2 = 0.1*omgR
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
lam = 10.0
L96 = 9.6*RJ  # Europa公転軌道 L値

# 木星とtrace座標系原点の距離(x軸の定義)
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
eurz = L96*math.sin(math.radians(lam))

# 遠方しきい値(z方向) 磁気緯度で設定
z_p_rad = math.radians(11.0)      # 北側
z_p = R0*math.cos(z_p_rad)**2 * math.sin(z_p_rad)
z_m_rad = math.radians(2.0)      # 南側
z_m = -R0*math.cos(z_m_rad)**2 * math.sin(z_m_rad)

# Europa真下(からさらに5m下)の座標と磁気緯度
z_below = eurz - RE - 5
z_below_rad = math.asin(
    (z_below)/math.sqrt((eurx + R0)**2 + eury**2 + (z_below)**2))

# DEPLETION領域
depletionR = 1.05*RE  # 円筒の半径
mphi_leading = math.atan2(eury+R0y+depletionR, eurx+R0x)     # 先行半球中心の磁気経度
mphi_trailing = math.atan2(eury+R0y-depletionR, eurx+R0x)    # 後行半球中心の磁気経度


#
#
# %% 初期位置エリア(z=0)での速度ベクトル (つまり磁気赤道面でのピッチ角)
v0eq = math.sqrt((energy/me)*2*float(1.602E-19))
V0 = v0eq
v0array = v0eq*np.ones(alphaeq.shape)

# ループの変数... 速度ベクトルとピッチ角
v0args = list(zip(
    list(v0array.reshape(v0array.size)),
    list(alphaeq.reshape(alphaeq.size))
))  # np.arrayは不可


#
#
# %% 初期座標ビンの設定ファンクション(磁気赤道面の2次元極座標 + z軸)
def init_points(rmin, rmax, phiJmin, phiJmax, zmin, zmax, nr, nphiJ, nz):
    r_bins = np.linspace(rmin, rmax, nr)
    phiJ_bins = np.radians(np.linspace(
        phiJmin, phiJmax, nphiJ))    # DEGREES TO RADIANS
    z_bins = np.array([0])

    # 動径方向の中点
    r_bins = np.linspace(rmin, rmax, 10)
    r_mae = r_bins[:-1]
    r_ushiro = r_bins[1:]
    r_middle = (r_ushiro + r_mae)*0.5   # r 中点

    # 経度方向の中点
    phiJ_mae = phiJ_bins[:-1]
    phiJ_ushiro = phiJ_bins[1:]
    phiJ_middle = (phiJ_ushiro + phiJ_mae)*0.5  # phiJ 中点

    # 3次元グリッドの作成
    r_grid, phiJ_grid, z_grid = np.meshgrid(r_middle, phiJ_middle, z_bins)

    # ビンの中でシフトさせる距離(r-phiJ平面)
    d_r = 0.5*np.abs(r_grid[0, 1, 0] - r_grid[0, 0, 0])
    d_phiJ = 0.5*np.abs(phiJ_grid[1, 0, 0] - phiJ_grid[0, 0, 0])

    return r_grid, phiJ_grid, z_grid, d_r, d_phiJ


# %% 初期座標ビンの設定(グローバル)
nr, nphiJ, nz = 20, 100, 1     # x, y, zのビン数
r_grid, phiJ_grid, z_grid, d_r, d_phiJ = init_points(r_im, r_ip,
                                                     # phiJの範囲(DEGREES)
                                                     -30.0, -1.0,
                                                     -1, 1,
                                                     nr, nphiJ, nz)


#
#
# %% 初期座標をシフトさせる
@jit('Tuple((f8,f8))(f8,f8)', nopython=True, fastmath=True)
def init_shift(r0, phiJ0):
    """
    DESCRIPTION IS HERE.
    """
    # ビンの中心からのずれ量 shapeは(ny,nx)
    r_shift = d_r*(2*np.random.rand() - 1)
    phiJ_shift = d_phiJ*(2*np.random.rand() - 1)

    # ビンの中心からずらした新しい座標
    r0 += r_shift
    phiJ0 += phiJ_shift

    x0 = r0*np.cos(phiJ0) - R0x
    y0 = r0*np.sin(phiJ0) - R0y

    return x0, y0


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
    DESCRIPTION IS HERE.
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
# %% 磁場ベクトル
@jit('f8[:](f8[:])', nopython=True, fastmath=True)
def Bfield(Rvec):
    """
    DESCRIPTION IS HERE.
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
# %% 磁場強度(スカラー)
@jit('f8(f8[:])', nopython=True, fastmath=True)
def Babs(Rvec):
    """
    DESCRIPTION IS HERE.
    """
    # x, y, zは木星からの距離
    Bvec = Bfield(Rvec)
    B = math.sqrt(Bvec[0]**2 + Bvec[1]**2 + Bvec[2]**2)

    return B


#
#
# %% 共回転電場ベクトル
def Efield(Rvec):
    """
    DESCRIPTION IS HERE.
    """
    # x, y, zは木星からの距離
    Bvec = Bfield(Rvec)
    Evec = np.dot(omgRvec, Bvec)*Rvec - np.dot(Rvec, Bvec)*omgRvec

    return Evec


#
#
# %% Newton法でミラーポイントの磁気緯度を調べる(ダイポール磁場)
@jit('f8(f8)', nopython=True, fastmath=True)
def mirror(alpha):
    """
    DESCRIPTION IS HERE.
    """
    xn = math.radians(1E-5)

    # ニュートン法の反復
    for _ in range(50):
        f = math.cos(xn)**6 - math.sqrt(1+3*math.sin(xn)**2) * \
            (math.sin(alpha)**2)
        fdash = -6*(math.cos(xn)**5)*math.sin(xn) - 3*(math.sqrt(1+3*math.sin(xn)
                                                                 ** 2)**(-1))*math.sin(xn)*math.cos(xn)*(math.sin(alpha)**2)
        xn += - f/fdash

    # xnは 360度以上 の数字になりうるので, 磁気緯度として相応しい値に変換する
    la = xn % (2*np.pi)
    if (la > 0.5*np.pi) and (la <= np.pi):
        la = np.pi - la
    elif (la > np.pi) and (la <= 1.5*np.pi):
        la += -np.pi
    elif (la > 1.5*np.pi) and (la < 2*np.pi):
        la = 2*np.pi - la

    print('mirror point: ', np.degrees(la))

    return la


#
#
# %% Newton法でミラーポイントの磁気緯度を調べる(ダイポール磁場)
@jit('f8(f8, f8)', nopython=True, fastmath=True)
def mirrorpoint(lamu, alphau):
    """
    DESCRIPTION IS HERE.
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

    print('mirror point: ', np.degrees(la))

    return la


#
#
# %% 共回転ドリフト速度
@jit('f8[:](f8,f8[:])', nopython=True, fastmath=True)
def Vdvector(omg, Rvec):
    """
    DESCRIPTION IS HERE.
    """
    omgvec = omg*eomg
    Vdvec = np.array([
        omgvec[1]*Rvec[2] - omgvec[2]*Rvec[1],
        omgvec[2]*Rvec[0] - omgvec[0]*Rvec[2],
        omgvec[0]*Rvec[1] - omgvec[1]*Rvec[0]
    ])

    return Vdvec


#
#
# %% 遠心力項
@jit('f8[:](f8,f8[:])', nopython=True, fastmath=True)
def centrif(omg, Rvec):
    """
    omgvec x (omgvec x Rvec)
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
# %% 自転軸からの距離 rho
@jit('f8(f8[:])', nopython=True, fastmath=True)
def Rho(Rvec):
    """
    DESCRIPTION IS HERE.
    """
    Rlen2 = Rvec[0]**2 + Rvec[1]**2 + Rvec[2]**2
    Rdot = eomg[0]*Rvec[0] + eomg[1]*Rvec[1] + eomg[2]*Rvec[2]
    rho = math.sqrt(Rlen2 - Rdot**2)

    return rho


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

    # DEPLETION領域
    mphiR = math.atan2(Rvec[1], Rvec[0])
    if (mphiR < mphi_leading) & (mphiR > mphi_trailing):   # IN THE DEPLETION REGION
        omg = omgR2
        print('omgR2')
    else:   # OUT OF THE DEPLETION REGION
        omg = omgR
        print('omgR')

    bvec = Bfield(Rvec)/Babs(Rvec)
    Vdvec = Vdvector(omg, Rvec)
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
    mphiR = math.atan2(Rvec_new[1], Rvec_new[0])
    if (mphiR < mphi_leading) & (mphiR > mphi_trailing):   # IN THE DEPLETION REGION
        omg = omgR2
    else:   # OUT OF THE DEPLETION REGION
        omg = omgR

    # 保存量
    bvec = Bfield(Rvec)/Babs(Rvec)
    Vdvec = Vdvector(omg, Rvec)
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
# %% ジャイロをフルに解く運動方程式(DEPLETION領域なので遠心力は省略...)
@jit('f8[:](f8[:],f8)', nopython=True, fastmath=True)
def eom(rv, t):
    """
    `rv` ... <ndarray> trace座標系 \\
    `t` ... 時刻 \\
    `K0` ... 保存量
    """
    # rv.shape >>> (6,)
    # 座標系 = Europa中心の静止系
    # rv[0] ... x of gyro motion
    # rv[1] ... y
    # rv[2] ... z
    # rv[3] ... vx
    # rv[4] ... vy
    # rv[5] ... vz

    # 木星原点の位置ベクトルに変換
    rvec = rv[0:3] + R0vec  # 座標
    vvec = rv[3:6]          # 速度

    # Magnetic Field
    Bvec = Bfield(rvec)   # 磁場ベクトル

    # ローレンツ力
    Lorentz = A1*np.array([
        vvec[1]*Bvec[2] - vvec[2]*Bvec[1],
        vvec[2]*Bvec[0] - vvec[0]*Bvec[2],
        vvec[0]*Bvec[1] - vvec[1]*Bvec[0]
    ])

    # 共回転速度ベクトル
    vdvec = Vdvector(omgR2, rvec)

    # 運動方程式array
    eom = np.array([
        vvec[0]+vdvec[0],
        vvec[1]+vdvec[1],
        vvec[2]+vdvec[2],
        Lorentz[0],
        Lorentz[1],
        Lorentz[2],
    ])

    return eom


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

    # DEPLETION領域
    mphiR = math.atan2(Rvec[1], Rvec[0])
    if (mphiR < mphi_leading) & (mphiR > mphi_trailing):   # IN THE DEPLETION REGION
        omg = omgR2

    else:   # OUT OF THE DEPLETION REGION
        omg = omgR

    # 遠心力項
    omgRxomgRxR_s = vecdot(bvec, -centrif(omg, Rvec))  # 遠心力項の平行成分

    # 共回転ドリフト速度
    Vdvec = Vdvector(omg, Rvec)          # 共回転ドリフト速度ベクトル
    Vdpara = vecdot(bvec, Vdvec)    # 平行成分

    # 微分の平行成分
    dVdparads = vecdot(
        bvec,
        np.array([
            (vecdot(bvec,
                    Vdvector(omg, Rvec+np.array([dX, 0., 0.]))) - Vdpara)/dX,
            (vecdot(bvec,
                    Vdvector(omg, Rvec+np.array([0., dY, 0.]))) - Vdpara)/dY,
            (vecdot(bvec,
                    Vdvector(omg, Rvec+np.array([0., 0., dZ]))) - Vdpara)/dZ
        ])
    )

    # 係数 mu
    mu = (K0-0.5*me*(RV[3]-Vdpara)**2 + 0.5*me*(Rho(Rvec)*omg)**2)/B

    # parallel速度の微分方程式
    dVparadt = -(mu/me)*dBds + omgRxomgRxR_s + (RV[3]-Vdpara)*dVdparads

    RVnew = np.array([Vparavec[0]+Vdvec[0], Vparavec[1]+Vdvec[1], Vparavec[2]+Vdvec[2],
                      dVparadt], dtype=np.float64)

    return RVnew


#
#
# %% 4次ルンゲクッタ.. functionの定義
# @jit('f8[:](f8[:],f8, f8)', nopython=True, fastmath=True)
def rk4(rv0, tsize, TC):
    """
    `rv0` ... <ndarray> trace座標系 \\
    `tsize` ... 時刻tのサイズ \\
    `TC` ... サイクロトロン周期 [s] \\
    Details follow: \\
    `RV0.shape` ... (7,) \\
    `rv0[0]` ... x of Guiding Center \\
    `rv0[1]` ... y \\
    `rv0[2]` ... z \\
    `rv0[3]` ... vx \\
    `rv0[4]` ... vy \\
    `rv0[5]` ... vz \\
    `rv0[6]` ... K0 (保存量)
    """

    # 時刻初期化
    t = 0

    # K0 保存量
    K0 = rv0[6]

    # 木星原点の位置ベクトルに変換
    rvec = rv0[0:3] + R0vec

    # ジャイロ半径
    # rgyro = me*math.sqrt(rv0[3]**2 + rv0[4]**2 + rv0[5]**2)/(-e*Babs(rvec))

    # ダイポールからの距離(@磁気赤道面 近似)
    r = math.sqrt(rvec[0]**2 + rvec[1]**2 + rvec[2]**2)
    req = r/(math.cos(lam)**2)

    # トレース開始
    rv = rv0[0:6]
    dt = FORWARD_BACKWARD*(6E-7)
    dt2 = 0.5*dt

    print('velocity: ', rv0[3:6])

    # 座標配列
    # trace[:, 0] ... x座標
    # trace[:, 1] ... y座標
    # trace[:, 2] ... z座標
    # trace[:, 3] ... v_parallel
    # trace[:, 4] ... K0
    trace = np.zeros((int(tsize/h), 5))
    kk = 0

    yn = 1

    # ジャイロを解くルンゲクッタ
    k = 0   # INDEX INITIALIZED
    for k in range(tsize-1):
        f1 = eom(rv, t)
        f2 = eom(rv+dt2*f1, t+dt2)
        f3 = eom(rv+dt2*f2, t+dt2)
        f4 = eom(rv+dt*f3, t+dt)
        rv2 = rv + dt*(f1 + 2*f2 + 2*f3 + f4)/6
        t += dt

        # Europaに再衝突
        r_eur0 = math.sqrt((rv2[0]-eurx)**2 + (rv2[1]-eury)
                           ** 2 + (rv2[2]-eurz)**2)
        print((rv2[2]-eurz)/1000)

        if r_eur0 < RE:
            print('Gyro Collision')
            yn = 0
            break

        rv = rv2

        # Europaの近く(60km上空)
        if r_eur0 < (RE+6E+4):
            continue
        else:
            print('Gyro Ended')
            break

    # 速度ベクトル V0vec
    V0vec = rv[3:6]

    # 磁場と平行な単位ベクトル
    rvec = rv[0:3]
    B = Babs(rvec + R0vec)
    bvec = Bfield(rvec + R0vec)/B

    # 自転軸からの距離 rho (BACKTRACING)
    rho = Rho(rvec + R0vec)
    Vdvec = Vdvector(omgR2, rvec + R0vec)
    Vdpara = bvec[0]*Vdvec[0] + bvec[1]*Vdvec[1] + bvec[2]*Vdvec[2]  # 平行成分

    # Gyro Period
    TC = 2*np.pi*me/(-e*B)
    dt = FORWARD_BACKWARD*25*TC
    dt2 = 0.5*dt

    # 速度ベクトルを分解
    vparallel = vecdot(bvec, V0vec)
    vperp = math.sqrt(V0**2 - vparallel**2)
    print('vparallel 2: ', vparallel)

    # 保存量 K0 (運動エネルギー)
    K0 = 0.5*me*((vparallel-Vdpara)**2 - (rho*omgR2)**2 + vperp**2)

    # 次のループへの変数
    RV0vec = np.array([
        rvec[0], rvec[1], rvec[2], vparallel, K0
    ])

    # 回転中心座標
    RV = RV0vec[0:4]

    # ルンゲクッタ
    print('Center Start')
    for k in range(tsize-1):
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
            # break

        # 木星に衝突
        r_jovi = math.sqrt(Rvec[0]**2 + Rvec[1]**2 + Rvec[2]**2)
        if r_jovi < RJ:
            print('Loss')
            break

        if k % h == 0:
            # 1ステップでどれくらい進んだか
            # D = np.linalg.norm(RV2[0:3]-RV[0:3])
            # print(t, D)
            trace[kk, :] = np.array([
                RV2[0], RV2[1], RV2[2], RV2[3], K0
            ])
            kk += 1

        if (RV[2] < z_p) and (RV2[2] > z_p):
            print('UPPER')
            RV2 = comeback(RV2, req, z_p_rad, K0)

        if (RV[2] > z_m) and (RV2[2] < z_m):
            print('LOWER')
            RV2 = comeback(RV2, req, z_m_rad, K0)

        # 磁気赤道面への到達
        # if (RV[2] < 0) and (RV2[2] > 0):
        #     print('South to Equator: ', t, 'sec')
        #     break

        if (RV[2] > 0) and (RV2[2] < 0):
            print('North to Equator: ', t,  'sec')
            break

        # Gyro period
        TC = 2*np.pi*me/(-e*Babs(Rvec))
        # print('TC: ', TC)

        # 時間刻みの更新
        dt = FORWARD_BACKWARD*20*TC
        dt2 = 0.5*dt

        # 座標と時刻更新
        RV = RV2
        t += dt

        if abs(t) > 5000:
            break

    # print('t: ', t)

    # DEPLETION領域かどうか
    mphiR = math.atan2(Rvec[1], Rvec[0])
    # IN THE DEPLETION REGION
    if (mphiR < mphi_leading) & (mphiR > mphi_trailing):
        omg = omgR2
    # OUT OF THE DEPLETION REGION
    else:
        omg = omgR

    bvec = Bfield(Rvec)/Babs(Rvec)
    Vdvec = Vdvector(omg, Rvec)
    Vdpara = vecdot(bvec, Vdvec)
    Vdperp = math.sqrt(Vdvec[0]**2 + Vdvec[1]
                       ** 2 + Vdvec[2]**2 - Vdpara**2)
    vperp = math.sqrt(
        2*K0/me - (RV[3]-Vdpara)**2 + (Rho(Rvec)*omgR)**2) - Vdperp
    vparallel = RV[3] - Vdpara
    Vnorm = math.sqrt(vparallel**2 + vperp**2)
    alpha_end = math.degrees(math.atan2(vperp, -vparallel))
    energy_end = me*0.5*(Vnorm**2)/float(1.602E-19)

    K1 = 0.5*me*((vparallel)**2 -
                 (Rho(Rvec)*omgR)**2 + (vperp+Vdperp)**2)
    print('alpha_end [degrees]: ', alpha_end)
    print('energy_end [eV]: ', energy_end)
    print('K1/K0: ', K1/K0)
    print('yn: ', yn)

    return trace[0:kk, :]


#
#
# %% 4次ルンゲクッタ.. classの定義
class RK4:
    def __init__(self, RV0, tsize, TC):
        result = rk4(RV0, tsize, TC)
        self.positions = result


#
#
# %% トレースを行うfunction
def calc(r0, phiJ0, z0):
    """
    DESCRIPTION IS HERE.
    """
    start0 = time.time()
    # 終点座標(x,y,z)と磁気赤道面ピッチ角(aeq)を格納する配列
    result = np.zeros((len(v0args), 4))
    i = 0
    for veq, aeq in v0args:
        dt = abs(1/(veq*math.cos(aeq))) * 100

        # 初期座標をシフトさせる
        x0, y0 = init_shift(r0, phiJ0)
        xv = np.array([
            x0,
            y0,
            z0
        ], dtype=np.float64)

        rk4 = RK4(xv, t, dt, tsize, veq, aeq)
        result[i, :] = rk4.positions
        i += 1
    print('A BIN DONE: %.3f seconds ----------' % (time.time() - start0))
    return result


#
#
# %% 時間設定
t = 0
dt = float(1E-5)  # 時間刻みはEuropaの近くまで来たらもっと細かくして、衝突判定の精度を上げよう
t_len = 5000
tsize = int(t_len/dt)


#
#
# %% main関数
def main():
    # 初期座標
    # r01 = r_grid.reshape(r_grid.size)  # 1次元化
    # phiJ01 = phiJ_grid.reshape(phiJ_grid.size)  # 1次元化
    # z01 = z_grid.reshape(z_grid.size)  # 1次元化

    # 初期座標
    # x01 = r01[0]*math.cos(phiJ01[0]) - R0x
    # y01 = r01[0]*math.sin(phiJ01[0]) - R0y
    # z01 = z01[0]
    # BACKTRACING
    x01 = eurx
    y01 = eury
    z01 = eurz + RE + 1E+4
    Rinitvec = np.array([x01, y01, z01], dtype=np.float64)
    # print(Rinitvec)

    # 初期速度ベクトル
    V0 = v0eq    # 単位: m/s

    # 速度ベクトル V0vec
    beta = 2*np.pi*np.random.rand()
    V0vec = V0*np.array([
        math.sin(alphaeq)*math.cos(beta),
        math.sin(alphaeq)*math.sin(beta),
        math.cos(alphaeq)
    ])

    # 第一断熱不変量
    Rvec = Rinitvec + R0vec
    B = Babs(Rvec)
    bvec = Bfield(Rvec)/B
    vparallel = vecdot(bvec, V0vec)
    vperp = math.sqrt(V0**2 - vparallel**2)
    print('B: ', B)
    print('vparallel: ', vparallel)
    mu0 = 0.5*me*(vperp**2)/B
    print('mu0: ', mu0)

    # 自転軸からの距離 rho (BACKTRACING)
    rho = Rho(Rvec)
    Vdvec = Vdvector(omgR2, Rvec)
    # Vd = math.sqrt(Vdvec[0]**2 + Vdvec[1]**2 + Vdvec[2]**2)
    Vdpara = bvec[0]*Vdvec[0] + bvec[1]*Vdvec[1] + bvec[2]*Vdvec[2]  # 平行成分
    K0 = 0.5*me*((vparallel - Vdpara)**2 - (rho*omgR2)**2 + vperp**2)
    mu = (K0 - 0.5*me*(vparallel-Vdpara)**2 + 0.5*me*(rho*omgR2)**2)/B
    # K0 = mu
    print('mu/mu0: ', mu/mu0)

    # Gyro Period
    TC = 2*np.pi*me/(-e*B)

    # 初期座標&初期速度ベクトルの配列
    # RV0vec[0] ... x座標
    # RV0vec[1] ... y座標
    # RV0vec[2] ... z座標
    # RV0vec[3] ... v parallel
    # RV0vec[4] ... K0 (保存量)
    rv0vec = np.array([
        Rinitvec[0],
        Rinitvec[1],
        Rinitvec[2],
        V0vec[0],
        V0vec[1],
        V0vec[2],
        K0
    ])

    # FORWARD
    if FORWARD_BACKWARD == 1:
        print('FORWARD START')
        start = time.time()
        forward_result = RK4(rv0vec, tsize, TC).positions
        print('%.3f seconds' % (time.time()-start))
        np.savetxt(
            '/Users/shin/Documents/Research/Europa/Codes/gyrocenter/gyrocenter_1/' +
            str(savename_f), forward_result
        )
        print('FORWARD DONE')

    # BACKWARD
    elif FORWARD_BACKWARD == -1:
        print('BACKWARD START')
        start = time.time()
        backward_result = RK4(rv0vec, tsize, TC).positions
        print('%.3f seconds' % (time.time()-start))
        np.savetxt(
            '/Users/shin/Documents/Research/Europa/Codes/gyrocenter/gyrocenter_1/' +
            str(savename_b), backward_result
        )
        print('BACKWARD DONE')

    return 0


#
#
# %% EXECUTE
if __name__ == '__main__':
    a = main()
