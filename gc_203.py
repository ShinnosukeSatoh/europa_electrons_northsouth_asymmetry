""" gc_203.py

Created on Sun Dec 27 16:07:00 2021
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
from numba import jit, f8, objmode
from numba.experimental import jitclass
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
# %% 座標保存の間隔(hステップに1回保存する)
h = int(500)


#
#
# %% SETTINGS FOR THE NEXT EXECUTION
energy = 10  # eV
savename_f = 'go_100ev_aeq60_20211225_1_forward.txt'
savename_b = 'go_100ev_aeq60_20211225_1_backward.txt'


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
omgR2 = 0.4*omgR
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
eurz = L96*math.sin(math.radians(lam)) - R0z

# 遠方しきい値(z方向) 磁気緯度で設定
z_p_rad = math.radians(11.0)      # 北側
z_p = R0*math.cos(z_p_rad)**2 * math.sin(z_p_rad)
z_m_rad = math.radians(11.0)      # 南側
z_m = -R0*math.cos(z_m_rad)**2 * math.sin(z_m_rad)


# DEPLETION領域
depletionR = 1.05*RE  # 円筒の半径
mphi_leading = math.atan2(eury+R0y+depletionR, eurx+R0x)     # 先行半球中心の磁気経度
mphi_trailing = math.atan2(eury+R0y-depletionR, eurx+R0x)    # 後行半球中心の磁気経度


#
#
# %% 初期位置エリア(z=0)での速度ベクトル (つまり磁気赤道面でのピッチ角)
V0 = math.sqrt((energy/me)*2*float(1.602E-19))
pitch = int(30)  # 0-90度を何分割するか
alphaeq0 = np.radians(np.linspace(0.1, 89.91, int(pitch+1)))   # PITCH ANGLE
a0c = (alphaeq0[1:] + alphaeq0[:-1])/2
alphaeq1 = np.radians(np.linspace(90.09, 179.9, int(pitch+1)))   # PITCH ANGLE
a1c = (alphaeq1[1:] + alphaeq1[:-1])/2
alpha_array = np.hstack([a0c, a1c])
# alpha_array = np.radians(np.linspace(89.9, 90.1, 4))


#
#
# %% Europa表面の点から放射状に粒子を放つときの初期条件の設定
def surface_init(colat_mesh, phi_mesh, v0, alpha, beta):
    Rinitvec = np.array([
        RE*np.sin(np.radians(colat_mesh)) *
        np.cos(np.radians(phi_mesh)),     # x0
        RE*np.sin(np.radians(colat_mesh)) * \
        np.sin(np.radians(phi_mesh)),     # y0
        # z0
        RE*np.cos(np.radians(colat_mesh))
    ])

    return Rinitvec


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
# %% 磁場
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
# %% Newton法でミラーポイントの磁気緯度を調べる
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
# %% 共回転ドリフト速度
@jit('f8[:](f8[:])', nopython=True, fastmath=True)
def Vdvector(Rvec):
    """
    DESCRIPTION IS HERE.
    """
    Vdvec = np.array([
        omgRvec[1]*Rvec[2] - omgRvec[2]*Rvec[1],
        omgRvec[2]*Rvec[0] - omgRvec[0]*Rvec[2],
        omgRvec[0]*Rvec[1] - omgRvec[1]*Rvec[0]
    ])

    return Vdvec


#
#
# %% 共回転ドリフト速度(IN THE DEPLETION REGION)
@jit('f8[:](f8[:])', nopython=True, fastmath=True)
def Vdvector2(Rvec):
    """
    DESCRIPTION IS HERE.
    """
    Vdvec = np.array([
        omgR2vec[1]*Rvec[2] - omgR2vec[2]*Rvec[1],
        omgR2vec[2]*Rvec[0] - omgR2vec[0]*Rvec[2],
        omgR2vec[0]*Rvec[1] - omgR2vec[1]*Rvec[0]
    ])

    return Vdvec


#
#
# %% 遠心力項
@jit('f8[:](f8[:])', nopython=True, fastmath=True)
def centrif(Rvec):
    """
    omgRvec x (omgRvec x Rvec)
    """
    cross3 = np.array([
        omgRvec[1]*(omgRvec[0]*Rvec[1]-omgRvec[1]*Rvec[0]) -
        omgRvec[2]*(omgRvec[2]*Rvec[0]-omgRvec[0]*Rvec[2]),
        omgRvec[2]*(omgRvec[1]*Rvec[2]-omgRvec[2]*Rvec[1]) -
        omgRvec[0]*(omgRvec[0]*Rvec[1]-omgRvec[1]*Rvec[0]),
        omgRvec[0]*(omgRvec[2]*Rvec[0]-omgRvec[0]*Rvec[2]) -
        omgRvec[1]*(omgRvec[1]*Rvec[2]-omgRvec[2]*Rvec[1])
    ])

    return cross3


#
#
# %% 遠心力項(IN THE DEPLETION REGION)
@jit('f8[:](f8[:])', nopython=True, fastmath=True)
def centrif2(Rvec):
    """
    omgR2vec x (omgRvec x Rvec)
    """
    cross3 = np.array([
        omgR2vec[1]*(omgR2vec[0]*Rvec[1]-omgR2vec[1]*Rvec[0]) -
        omgR2vec[2]*(omgR2vec[2]*Rvec[0]-omgR2vec[0]*Rvec[2]),
        omgR2vec[2]*(omgR2vec[1]*Rvec[2]-omgR2vec[2]*Rvec[1]) -
        omgR2vec[0]*(omgR2vec[0]*Rvec[1]-omgR2vec[1]*Rvec[0]),
        omgR2vec[0]*(omgR2vec[2]*Rvec[0]-omgR2vec[0]*Rvec[2]) -
        omgR2vec[1]*(omgR2vec[1]*Rvec[2]-omgR2vec[2]*Rvec[1])
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
@jit('f8[:](f8[:],f8,f8,f8,f8)', nopython=True, fastmath=True)
def comeback(RV2, req, lam0, mirlam, K0):
    """
    DESCRIPTION IS HERE.

    """
    # Rvec: 木星原点
    # lam0: スタートの磁気緯度
    # mirlam: mirror pointの磁気緯度

    Rvec = RV2[0:3] + R0vec
    bvec = Bfield(Rvec)/Babs(Rvec)
    Vdvec = Vdvector(Rvec)
    Vdpara = vecdot(bvec, Vdvec)
    Vdperp = math.sqrt(Vdvec[0]**2 + Vdvec[1]
                       ** 2 + Vdvec[2]**2 - Vdpara**2)
    vperp = math.sqrt(
        2*K0/me - (RV2[3]-Vdpara)**2 + (Rho(Rvec)*omgR)**2) - Vdperp
    vparallel = RV2[3] - Vdpara
    v_new = math.sqrt(vparallel**2 + vperp**2)
    veq = v_new

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
    Rvec_new = Corotation(Rvec, omgR*tau)

    # 保存量
    bvec = Bfield(Rvec)/Babs(Rvec)
    Vdvec = Vdvector(Rvec)
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
    DESCRIPTION IS HERE.
    `aaaa`
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
        # 遠心力項
        omgRxomgRxR_s = vecdot(bvec, -centrif2(Rvec))  # 遠心力項の平行成分

        # 共回転ドリフト速度
        Vdvec = Vdvector2(Rvec)          # 共回転ドリフト速度ベクトル
        Vdpara = vecdot(bvec, Vdvec)    # 平行成分

        omg = omgR2

        # 微分の平行成分
        dVdparads = vecdot(
            bvec,
            np.array([
                (vecdot(bvec,
                        Vdvector2(Rvec+np.array([dX, 0., 0.]))) - Vdpara)/dX,
                (vecdot(bvec,
                        Vdvector2(Rvec+np.array([0., dY, 0.]))) - Vdpara)/dY,
                (vecdot(bvec,
                        Vdvector2(Rvec+np.array([0., 0., dZ]))) - Vdpara)/dZ
            ])
        )

    else:   # OUT OF THE DEPLETION REGION
        # 遠心力項
        omgRxomgRxR_s = vecdot(bvec, -centrif(Rvec))  # 遠心力項の平行成分

        # 共回転ドリフト速度
        Vdvec = Vdvector(Rvec)          # 共回転ドリフト速度ベクトル
        Vdpara = vecdot(bvec, Vdvec)    # 平行成分

        omg = omgR

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

    RVnew = np.array([Vparavec[0]+Vdvec[0], Vparavec[1]+Vdvec[1], Vparavec[2]+Vdvec[2],
                      dVparadt], dtype=np.float64)

    return RVnew


#
#
# %% 4次ルンゲクッタ.. functionの定義
@jit('f8[:](f8[:],f8, f8)', nopython=True, fastmath=True)
def rk4(RV0, tsize, TC):
    """
    DESCRIPTION IS HERE.
    """
    # RV0.shape >>> (6,)
    # 座標系 = Europa中心の静止系
    # RV0[0] ... x of Guiding Center
    # RV0[1] ... y
    # RV0[2] ... z
    # RV0[3] ... v parallel
    # RV0[4] ... K0 (保存量)
    # aeq: RADIANS

    # 時刻初期化
    t = 0

    # K0 保存量
    K0 = RV0[4]

    # 木星原点の位置ベクトルに変換
    Rvec = RV0[0:3] + R0vec

    # トレース開始
    RV = RV0[0:4]
    dt = FORWARD_BACKWARD*20*TC
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
            break

        # 木星に衝突
        r_jovi = math.sqrt(Rvec[0]**2 + Rvec[1]**2 + Rvec[2]**2)
        if r_jovi < RJ:
            yn = 0
            # print('Loss')
            break

        # 磁気赤道面への到達
        # if (RV[2] < 0) and (RV2[2] > 0):
        #     print('South to Equator: ', t, 'sec')
        #     break

        if (RV[2] > 0) and (RV2[2] < 0):
            Rvec = 0.5*(RV[0:3] + RV2[0:3]) + R0vec
            # print('North to Equator: ', t,  'sec')
            bvec = Bfield(Rvec)/Babs(Rvec)
            Vdvec = Vdvector(Rvec)
            Vdpara = vecdot(bvec, Vdvec)
            Vdperp = math.sqrt(Vdvec[0]**2 + Vdvec[1]
                               ** 2 + Vdvec[2]**2 - Vdpara**2)
            vperp = math.sqrt(
                2*K0/me - (RV[3]-Vdpara)**2 + (Rho(Rvec)*omgR)**2) - Vdperp
            vparallel = RV[3] - Vdpara
            Vnorm = math.sqrt(vparallel**2 + vperp**2)
            alpha_end = math.atan2(vperp, -vparallel)   # RADIANS
            energy_end = me*0.5*(Vnorm**2)/float(1.602E-19)

            K1 = 0.5*me*((vparallel)**2 -
                         (Rho(Rvec)*omgR)**2 + (vperp+Vdperp)**2)
            print('alpha_end [degrees]: ', math.degrees(alpha_end))
            print('energy_end [eV]: ', energy_end)
            print('K1/K0: ', K1/K0)

            # 座標配列
            # trace[0] ... 終点 x座標
            # trace[1] ... 終点 y座標
            # trace[2] ... 終点 z座標
            # trace[3] ... yn
            # trace[4] ... 終点 energy [eV]
            # trace[5] ... 終点 alpha_eq [RADIANS]
            trace[0:3] = RV[0:3]
            trace[3] = energy_end
            trace[4] = alpha_end
            trace[5] = yn

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

    Rinitvec = np.array([
        RE*math.sin(mcolatr)*math.cos(mlongr),
        RE*math.sin(mcolatr)*math.sin(mlongr),
        RE*math.cos(mcolatr)
    ])

    # 表面法線ベクトル
    nvec = Rinitvec / \
        math.sqrt(Rinitvec[0]**2 + Rinitvec[1]**2 + Rinitvec[2]**2)

    # TRACE座標系に
    Rinitvec = np.array([
        Rinitvec[0] + eurx,
        Rinitvec[1] + eury,
        Rinitvec[2] + eurz
    ])

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

    # 磁場と平行な単位ベクトル
    B = Babs(Rinitvec + R0vec)
    bvec = Bfield(Rinitvec + R0vec)/B

    # 自転軸からの距離 rho (BACKTRACING)
    rho = Rho(Rinitvec + R0vec)
    Vdvec = Vdvector2(Rinitvec + R0vec)
    Vdpara = bvec[0]*Vdvec[0] + bvec[1]*Vdvec[1] + bvec[2]*Vdvec[2]  # 平行成分

    # Gyro Period
    TC = 2*np.pi*me/(-e*B)

    # LOOP INDEX INITIALIZED
    i = 0
    for alpha in alpha_array:
        beta = 2*np.pi*np.random.rand()
        V0vec = V0*np.array([
            math.sin(alpha)*math.cos(beta),
            math.sin(alpha)*math.sin(beta),
            math.cos(alpha)
        ])
        vparallel = vecdot(bvec, V0vec)
        vperp = math.sqrt(V0**2 - vparallel**2)
        vdotn = vecdot(nvec, V0vec)

        # 自転軸からの距離 rho (BACKTRACING)
        K0 = 0.5*me*((vparallel - Vdpara)**2 - (rho*omgR2)**2 + vperp**2)

        # 初期座標&初期速度ベクトルの配列
        # RV0vec[0] ... x座標
        # RV0vec[1] ... y座標
        # RV0vec[2] ... z座標
        # RV0vec[3] ... v parallel
        # RV0vec[4] ... K0 (保存量)
        RV0vec = np.array([
            Rinitvec[0], Rinitvec[1], Rinitvec[2], vparallel, K0
        ])

        # TRACING
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

    with objmode():
        print('A BIN DONE [sec]: ',  (time.perf_counter() - start0))
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
    # Europa表面の初期座標
    ncolat = 50  # 分割数
    nphi = 100    # 分割数
    meshlong, meshcolat = np.meshgrid(
        np.radians(np.linspace(0, 360, nphi)),
        np.radians(np.linspace(0, 180, ncolat))
    )
    # meshlong: 経度
    # meshcolat: 余緯度

    mcolatr = meshcolat.reshape(meshcolat.size)  # 1次元化
    mlongr = meshlong.reshape(meshlong.size)  # 1次元化

    # 情報表示
    print('alpha: {:>7d}'.format(alpha_array.size))
    print('ncolat: {:>7d}'.format(ncolat))
    print('nphi: {:>7d}'.format(nphi))
    print('total: {:>7d}'.format(alpha_array.size*ncolat*nphi))
    # print(savename)

    # 並列計算用 変数リスト(zip)
    args = list(zip(mcolatr, mlongr))  # np.arrayは不可。ここが1次元なのでpoolの結果も1次元。

    # 並列計算の実行
    start = time.time()
    with Pool(processes=1) as pool:
        result_list = list(pool.starmap(calc, args))
    stop = time.time()
    print('%.3f seconds' % (stop - start))

    # 返り値(配列 *行8列)
    # 結果をreshape
    result = np.array(result_list)
    result = result.reshape([int(alpha_array.size*ncolat*nphi), 10])
    print(result.shape)
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
    # result[:, 9] ... 出発点 v_dot_n

    # Europaに衝突しない(yn=1)の粒子を取り出す
    yn1 = np.where(result[:, 6] == 1)  # 0でない行を見つける(検索は表面yn=6列目)
    mageq = result[yn1]  # 0でない行だけ取り出し

    """
    np.savetxt(
    '/Users/shin/Documents/Research/Europa/Codes/gyrocenter/gyrocenter_1/' +
    str(savename_f), trace
    )
    """

    # 情報表示
    print('alpha: {:>7d}'.format(alpha_array.size))
    print('ncolat: {:>7d}'.format(ncolat))
    print('nphi: {:>7d}'.format(nphi))
    print('total: {:>7d}'.format(alpha_array.size*ncolat*nphi))

    print('magnetic equator: {:>7d}'.format(mageq[:, 0].size))
    # print(savename)

    return 0


#
#
# %% EXECUTE
if __name__ == '__main__':
    a = main()
