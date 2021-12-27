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
from numba import jit, f8
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
omgRvec = np.array([0., 0., omgR], dtype=np.float64)        # ベクトル化 単位: rad/s
eomg = np.array([-math.sin(math.radians(10)),
                 0., math.cos(math.radians(10))], dtype=np.float64)
omgRvec = omgR*eomg


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
z_p_rad = math.radians(10)      # 北側
z_p = R0*math.cos(z_p_rad)**2 * math.sin(z_p_rad)
z_m_rad = math.radians(10.0)    # 南側
z_m = -R0*math.cos(z_m_rad)**2 * math.sin(z_m_rad)


# Europa真下(からさらに5m下)の座標と磁気緯度
z_below = eurz - RE - 5
z_below_rad = math.asin(
    (z_below)/math.sqrt((eurx + R0)**2 + eury**2 + (z_below)**2))


#
#
# %% 初期位置エリア(z=0)での速度ベクトル (つまり磁気赤道面でのピッチ角)
V0 = math.sqrt((energy/me)*2*float(1.602E-19))
pitch = int(2)  # 0-90度を何分割するか
alphaeq0 = np.radians(np.linspace(0.1, 89.91, int(pitch+1)))   # PITCH ANGLE
a0c = (alphaeq0[1:] + alphaeq0[:-1])/2
alphaeq1 = np.radians(np.linspace(90.09, 179.9, int(pitch+1)))   # PITCH ANGLE
a1c = (alphaeq1[1:] + alphaeq1[:-1])/2
alpha_array = np.hstack([a0c, a1c])


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
# %% 磁場
@jit('f8[:](f8[:])', nopython=True, fastmath=True)
def Bfield(Rvec):
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


@jit('f8(f8[:])', nopython=True, fastmath=True)
def Babs(Rvec):
    # x, y, zは木星からの距離
    Bvec = Bfield(Rvec)
    B = math.sqrt(Bvec[0]**2 + Bvec[1]**2 + Bvec[2]**2)

    return B


#
#
# %% 共回転電場
def Efield(Rvec):
    # x, y, zは木星からの距離
    Bvec = Bfield(Rvec)
    Evec = np.dot(omgRvec, Bvec)*Rvec - np.dot(Rvec, Bvec)*omgRvec

    return Evec


#
#
# %% Newton法でミラーポイントの磁気緯度を調べる
@jit('f8(f8)', nopython=True, fastmath=True)
def mirror(alpha):
    xn = math.radians(1E-6)

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
# %% シミュレーションボックスの外に出た粒子の復帰座標を計算
@jit('f8(f8[:],f8,f8,f8)', nopython=True, fastmath=True)
def comeback(RV, req, lam0, mirlam):
    # RV: Europa原点座標系
    # lam0: スタートの磁気緯度
    # mirlam: mirror pointの磁気緯度
    x = RV[0] + R0x
    y = RV[1] + R0y

    # 積分の刻み
    dellam = 1E-5

    # 積分の長さ
    lamlen = int((mirlam-lam0)/dellam)

    # 積分の台形近似
    tau0, tau1, tau2 = 0, 0, 0  # initialize
    for _ in range(lamlen):
        tau1 = math.cos(lam0) * math.sqrt(1+3*math.sin(lam0)**2) * \
            math.sqrt(1-((math.cos(mirlam) / math.cos(lam0))**6) *
                      math.sqrt((1+3*math.sin(lam0)**2)/(1+3*math.sin(mirlam)**2)))**(-1)
        lam0 += dellam
        tau2 = math.cos(lam0) * math.sqrt(1+3*math.sin(lam0)**2) * \
            math.sqrt(1-((math.cos(mirlam) / math.cos(lam0))**6) *
                      math.sqrt((1+3*math.sin(lam0)**2)/(1+3*math.sin(mirlam)**2)))**(-1)

        tau0 += (req/V0)*0.5*(tau1+tau2)*dellam

    # 共回転で流される距離(y方向)
    tau = 2*tau0
    xd = x*math.cos(omgR*tau) - y*math.sin(omgR*tau)
    yd = y*math.cos(omgR*tau) + x*math.sin(omgR*tau)

    # 復帰座標
    Rvec2 = np.array([xd - R0x,
                      yd - R0y,
                      RV[2]], dtype=np.float64)

    # 復帰座標における速度ベクトル
    # vzだけ反転させるのは誤り
    # x,y成分だけ回転させて、z成分は反転?
    # → ダイポール軸と木星自転軸が平行でない場合に適切ではない。

    # return np.array([Rvec2[0], Rvec2[1], Rvec2[2], RV[3], RV[4], RV[5]])
    return 0  # Rvec2


#
#
# %% 共回転ドリフト速度
@jit('f8[:](f8[:])', nopython=True, fastmath=True)
def Vdvector(Rvec):
    Vdvec = np.array([
        omgRvec[1]*Rvec[2] - omgRvec[2]*Rvec[1],
        omgRvec[2]*Rvec[0] - omgRvec[0]*Rvec[2],
        omgRvec[0]*Rvec[1] - omgRvec[1]*Rvec[0]
    ])

    return Vdvec


#
#
# %% 回転中心位置ベクトルRについての運動方程式(eq.8 on Northrop and Birmingham, 1982)
@jit('f8[:](f8[:],f8, f8)', nopython=True, fastmath=True)
def ode2(RV, t, mu0):
    # RV.shape >>> (6,)
    # 座標系 = Europa中心の静止系
    # RV[0] ... x of Guiding Center
    # RV[1] ... y
    # RV[2] ... z
    # RV[3] ... v parallel

    # 木星原点の位置ベクトルに変換
    Rvec = RV[0:3] + R0vec

    # Magnetic Field
    B = Babs(Rvec)       # 磁場強度
    Bvec = Bfield(Rvec)  # 磁場ベクトル
    bvec = Bvec/B        # 磁力線方向の単位ベクトル
    # print('bvec: ', bvec)

    # 磁場強度の磁力線に沿った微分
    dX = 10.
    dY = 10.
    dZ = 10.
    ds = 10.
    dBds = (Babs(Rvec+ds*bvec) - B)/ds

    # 共回転ドリフト速度
    Vdvec = Vdvector(Rvec)
    VdX = Vdvector(Rvec+np.array([dX, 0., 0.]))
    VdY = Vdvector(Rvec+np.array([0., dY, 0.]))
    VdZ = Vdvector(Rvec+np.array([0., 0., dZ]))
    Vdpara = bvec[0]*Vdvec[0] + bvec[1]*Vdvec[1] + bvec[2]*Vdvec[2]  # 平行成分
    VdparaX = bvec[0]*VdX[0] + bvec[1]*VdX[1] + bvec[2]*VdX[2]  # 平行成分
    VdparaY = bvec[0]*VdY[0] + bvec[1]*VdY[1] + bvec[2]*VdY[2]  # 平行成分
    VdparaZ = bvec[0]*VdZ[0] + bvec[1]*VdZ[1] + bvec[2]*VdZ[2]  # 平行成分
    dVdparadR = np.array([
        (VdparaX - Vdpara)/dX,
        (VdparaY - Vdpara)/dY,
        (VdparaZ - Vdpara)/dZ
    ])
    dVdparads = bvec[0]*dVdparadR[0] + bvec[1] * \
        dVdparadR[1] + bvec[2]*dVdparadR[2]

    # 磁力線に平行な速度
    Vparavec = RV[3]*bvec

    # 自転軸からの距離 rho
    Rlen2 = Rvec[0]**2 + Rvec[1]**2 + Rvec[2]**2
    Rdot = eomg[0]*Rvec[0] + eomg[1]*Rvec[1] + eomg[2]*Rvec[2]
    rho = math.sqrt(Rlen2 - Rdot**2)

    Rlen2X = (Rvec[0]+dX)**2 + Rvec[1]**2 + Rvec[2]**2
    RdotX = eomg[0]*(Rvec[0]+dX) + eomg[1]*Rvec[1] + eomg[2]*Rvec[2]
    rhoX = math.sqrt(Rlen2X - RdotX**2)

    Rlen2Y = Rvec[0]**2 + (Rvec[1]+dY)**2 + Rvec[2]**2
    RdotY = eomg[0]*Rvec[0] + eomg[1]*(Rvec[1]+dY) + eomg[2]*Rvec[2]
    rhoY = math.sqrt(Rlen2Y - RdotY**2)

    Rlen2Z = Rvec[0]**2 + Rvec[1]**2 + (Rvec[2]+dZ)**2
    RdotZ = eomg[0]*Rvec[0] + eomg[1]*Rvec[1] + eomg[2]*(Rvec[2]+dZ)
    rhoZ = math.sqrt(Rlen2Z - RdotZ**2)

    drhodR = np.array([
        (rhoX - rho)/dX,
        (rhoY - rho)/dY,
        (rhoZ - rho)/dZ
    ])
    drhods = bvec[0]*drhodR[0] + bvec[1]*drhodR[1] + bvec[2]*drhodR[2]

    dVparadt = -mu0*dBds + (omgR**2)*rho*drhods + (RV[3]-Vdpara)*dVdparads
    # print('rho:', rho/RJ)
    RVnew = np.array([Vparavec[0]+Vdvec[0], Vparavec[1]+Vdvec[1], Vparavec[2]+Vdvec[2],
                      dVparadt], dtype=np.float64)

    return RVnew


#
#
# %% 4次ルンゲクッタ.. functionの定義
@jit(nopython=True, fastmath=True)
def rk4(RV0, tsize, TC):
    # RV0.shape >>> (6,)
    # 座標系 = Europa中心の静止系
    # RV0[0] ... x of Guiding Center
    # RV0[1] ... y
    # RV0[2] ... z
    # RV0[3] ... v parallel
    # RV0[4] ... mu0
    # aeq: RADIANS

    # 時刻初期化
    t = 0

    # 磁気モーメント
    mu0 = RV0[4]

    # 木星原点の位置ベクトルに変換
    Rvec = RV0[0:3] + R0vec

    # トレース開始
    RV = RV0[0:4]
    dt = FORWARD_BACKWARD*20*TC
    dt2 = dt*0.5

    # 座標配列
    # trace[:, 0] ... x座標
    # trace[:, 1] ... y座標
    # trace[:, 2] ... z座標
    # trace[:, 3] ... v_parallel
    # trace[:, 4] ... mu0 (Bがわかればv_perpを取り出せる)

    # ルンゲクッタ
    print('RK4 START')
    yn = 1
    for k in range(tsize-1):
        F1 = ode2(RV, t, mu0)
        F2 = ode2(RV+dt2*F1, t+dt2, mu0)
        F3 = ode2(RV+dt2*F2, t+dt2, mu0)
        F4 = ode2(RV+dt*F3, t+dt, mu0)
        RV2 = RV + dt*(F1 + 2*F2 + 2*F3 + F4)/6

        # 木星原点の位置ベクトルに変換
        Rvec = RV2[0:3] + R0vec

        # Gyro period
        TC = 2*np.pi*me/(-e*Babs(Rvec))

        eurR = math.sqrt((RV2[0]-eurx)**2 + (RV2[1]-eury)
                         ** 2 + (RV2[2]-eurz)**2)
        if (yn == 1) & (eurR < RE):
            yn = 0
            print('Collide')

        if (RV[3] > 0) & (RV2[3] < 0):
            print('Magnetic Equator')
            break

        # 時間刻みの更新
        dt = FORWARD_BACKWARD*20*TC
        dt2 = 0.5*dt

        # 座標更新
        RV = RV2
        t += dt

    # 第一断熱不変量
    B = Babs(RV[0:3] + R0vec)    # 磁気赤道面のはず(z=0)
    vperp = np.sqrt(2*mu0*B)
    vparallel = RV[3]

    # 座標配列
    # trace[:, 0] ... x座標
    # trace[:, 1] ... y座標
    # trace[:, 2] ... z座標
    # trace[:, 3] ... v_parallel
    # trace[:, 4] ... v_perp
    # trace[:, 5] ... yn
    trace = np.array([
        RV[0], RV[1], RV[2], vparallel, vperp, yn
    ])

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
def calc(colat, long):
    start0 = time.time()

    Rinitvec = np.array([
        RE*np.sin(colat)*np.cos(long) + eurx,
        RE*np.sin(colat)*np.sin(long) + eury,
        RE*np.cos(colat) + eurz
    ])

    # 出発点(余緯度, 経度)と終点座標(x,y,z)とv_parallel, v_perp, ynを格納する配列
    result = np.zeros((len(alpha_array), 8))

    i = 0
    for alpha in alpha_array:
        # 第一断熱不変量
        B = Babs(Rinitvec + R0vec)
        vparallel = V0*math.cos(alpha)
        vperp = V0*math.sin(alpha)
        mu0 = 0.5*(vperp**2)/B

        # Gyro Period
        TC = 2*np.pi*me/(-e*B)

        # 初期座標&初期速度ベクトルの配列
        # RV0vec[0] ... x座標
        # RV0vec[1] ... y座標
        # RV0vec[2] ... z座標
        # RV0vec[3] ... v parallel
        # RV0vec[4] ... mu0
        RV0vec = np.array([
            Rinitvec[0], Rinitvec[1], Rinitvec[2], vparallel, mu0
        ])

        rk4 = RK4(RV0vec, tsize, TC)

        # 初期座標&初期速度ベクトルの配列
        # result[:, 0] ... 出発点 余緯度
        # result[:, 1] ... 出発点 経度
        # result[:, 2] ... 終点 x座標
        # result[:, 3] ... 終点 y座標
        # result[:, 4] ... 終点 z座標
        # result[:, 5] ... v_parallel
        # result[:, 6] ... v_perp
        # result[:, 7] ... yn
        result[i, 0:2] = np.array([colat, long])
        result[i, 2:8] = rk4.positions
        i += 1
    print('A BIN DONE: %.3f seconds ----------' % (time.time() - start0))
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
    ncolat = 10  # 分割数
    nphi = 20    # 分割数
    colat_array = np.radians(np.linspace(0, 180, ncolat))
    long_array = np.radians(np.linspace(0, 360, nphi))
    meshlong, meshcolat = np.meshgrid(long_array, colat_array)
    # meshlong: 経度
    # meshcolat: 余緯度
    Rinitvec = np.array([
        RE*np.sin(meshcolat)*np.cos(meshlong) + eurx,
        RE*np.sin(meshcolat)*np.sin(meshlong) + eury,
        RE*np.cos(meshcolat) + eurz
    ])

    # x0r = Rinitvec[0].reshape(Rinitvec[0].size)  # 1次元化
    # y0r = Rinitvec[1].reshape(Rinitvec[1].size)  # 1次元化
    # z0r = Rinitvec[2].reshape(Rinitvec[2].size)  # 1次元化

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
    trace = np.array(result_list)
    trace = trace.reshape([int(alpha_array.size*ncolat*nphi), 8])
    print(trace.shape)
    # print(trace)

    # 初期座標&初期速度ベクトルの配列
    # trace[0] ... 出発点 余緯度
    # trace[1] ... 出発点 経度
    # trace[2] ... 終点 x座標
    # trace[3] ... 終点 y座標
    # trace[4] ... 終点 z座標
    # trace[5] ... v_parallel
    # trace[6] ... v_perp
    # trace[7] ... yn
    """
    np.savetxt(
    '/Users/shin/Documents/Research/Europa/Codes/gyrocenter/gyrocenter_1/' +
    str(savename_f), trace
    )
    """

    return 0


#
#
# %% EXECUTE
if __name__ == '__main__':
    a = main()
