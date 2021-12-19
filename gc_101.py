""" gc_101.py

Created on Sun Dec 19 16:34:00 2021
@author: Shin Satoh

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
FORWARD_BACKWARD = 1  # 1=FORWARD, -1=BACKWARD
h = 1

#
#
# %% SETTINGS FOR THE NEXT EXECUTION
energy = float(20)  # eV
savename = 'go_20ev_aeq77_20211125_3.txt'
alphaeq = np.radians(77)   # PITCH ANGLE


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
omgJ = FORWARD_BACKWARD*float(1.74E-4)    # 木星の自転角速度 単位: rad/s
omgE = FORWARD_BACKWARD*float(2.05E-5)    # Europaの公転角速度 単位: rad/s
omgR = omgJ-omgE        # 木星のEuropaに対する相対的な自転角速度 単位: rad/s


#
#
# %% 途中計算でよく出てくる定数の比
# A1 = float(e/me)             # 運動方程式内の定数
# A2 = float(mu*Mdip/4/3.14)  # ダイポール磁場表式内の定数
A1 = float(-1.7582E+11)    # 運動方程式内の定数
A2 = FORWARD_BACKWARD*1.60432E+20            # ダイポール磁場表式内の定数
A3 = 4*3.1415*me/(mu*Mdip*e)    # ドリフト速度の係数


#
#
# %% EUROPA POSITION (DETERMINED BY MAGNETIC LATITUDE)
lam = 0.0
L96 = 9.6*RJ  # Europa公転軌道 L値

# 木星とtrace座標系原点の距離(x軸の定義)
R0 = L96*(np.cos(np.radians(lam)))**(-2)
R0x = R0
R0y = 0
R0z = 0

# 初期条件座標エリアの範囲(木星磁気圏動径方向 最大と最小 phiJ=0で決める)
r_ip = (L96+1.15*RE)*(math.cos(math.radians(lam)))**(-2) - R0x
r_im = (L96-1.15*RE)*(math.cos(math.radians(lam)))**(-2) - R0x

# Europaのtrace座標系における位置
eurx = L96*math.cos(math.radians(lam)) - R0x
eury = 0  # ============================== ここ変えてね ==============================
eurz = L96*math.sin(math.radians(lam))

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
v0eq = math.sqrt((energy/me)*2*float(1.602E-19))
v0array = v0eq*np.ones(alphaeq.shape)

# ループの変数... 速度ベクトルとピッチ角
v0args = list(zip(
    list(v0array.reshape(v0array.size)),
    list(alphaeq.reshape(alphaeq.size))
))  # np.arrayは不可


#
#
# %% 初期座標ビンの設定ファンクション(磁気赤道面の2次元極座標 + z軸)
@jit(nopython=True, fastmath=True)
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
                                                     -5.0, -1.0,
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

    x0 = r0*np.cos(phiJ0)
    y0 = r0*np.sin(phiJ0)

    return x0, y0


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


@jit('f8(f8,f8,f8)', nopython=True, fastmath=True)
def Babs(x, y, z):
    # x, y, zは木星からの距離
    r = math.sqrt(x**2 + y**2 + z**2)
    B = A2 * (math.sqrt(1+3*(z/r))**2) * r**(-3)

    return B


# %% Newton法でミラーポイントの磁気緯度を調べる
@jit('f8(f8)', nopython=True, fastmath=True)
def mirror(alpha):
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

    print(np.degrees(la))

    return la


#
#
# %% シミュレーションボックスの外に出た粒子の復帰座標を計算
@jit('f8[:](f8[:],f8,f8,f8)', nopython=True, fastmath=True)
def comeback(xv, req, lam0, mirlam):
    # lam0: スタートの磁気緯度
    # mirlam: mirror pointの磁気緯度
    x = xv[0] + R0x
    y = xv[1] + R0y

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

        tau0 += (req/v0eq)*0.5*(tau1+tau2)*dellam

    # 共回転で流される距離(y方向)
    tau = 2*tau0
    xd = x*math.cos(omgR*tau) - y*math.sin(omgR*tau)
    yd = y*math.cos(omgR*tau) + x*math.sin(omgR*tau)

    # 復帰座標
    xv2 = np.array([xd - R0x,
                    yd - R0y,
                    xv[2]], dtype=np.float64)

    return xv2


#
#
# %% 電子位置ベクトルrについての一階微分方程式
@jit('f8[:](f8[:],f8,f8,f8,f8)', nopython=True, fastmath=True)
def ode(xyz, t, veq, aeq, Beq):
    x = xyz[0] + R0x
    y = xyz[1] + R0y
    z = xyz[2] + R0z

    r = math.sqrt(x**2 + y**2 + z**2)
    R = math.sqrt(x**2 + y**2)

    # Parallel
    nakami = 1 - r**5 * math.sin(aeq)**2 * \
        math.sqrt(r**2 + 3 * z**2) * (x**2 + y**2)**(-3)

    coef = veq * math.sqrt(nakami) * (r**(-1)) * (r**2 + 3 * z**2)**(-0.5)

    v_p_x = 3 * z * x
    v_p_y = 3 * z * y
    v_p_z = (2 * z**2) - (x**2 + y**2)

    v_par = coef * np.array([v_p_x, v_p_y, v_p_z], dtype=np.float64)
    vpa2 = (coef**2) * (v_p_x**2 + v_p_y**2 + v_p_z**2)

    # Drift
    vpe2 = (veq**2) * (Babs(x, y, z)/Beq) * math.sin(aeq)**2
    theta = math.acos(z/r)
    nakami = (vpa2) * r * math.sqrt(x**2 + y**2)
    nakami2 = 0.5 * vpe2 * (r**2) * math.sin(theta) * \
        (1 + (z/r)**2) * (1 + 3*(z/r)**2)**(-1)

    vb = A3 * (1+3*(z/r)**2)**(-1) * (nakami+nakami2)

    # 経度方向単位ベクトル
    e_phi = np.array([-y, x, 0], dtype=np.float64)/R

    # 経度方向ドリフト速度
    v_drift = (R*omgR + vb)*e_phi

    return v_par + v_drift


#
#
# %% 4次ルンゲクッタ.. functionの定義
@ jit(nopython=True, fastmath=True)
def rk4(xv, t, dt, tsize, veq, aeq):
    # xvはtrace座標系
    # aeq: RADIANS

    # 木星中心の座標
    x = xv[0] + R0x
    y = xv[1] + R0y
    z = xv[2] + R0z

    # 磁気赤道面での磁場
    Beq = Babs(x, y, z)

    # distance from Jupiter
    req = math.sqrt(x**2 + y**2 + z**2)

    # mirror pointの磁気緯度(rad)
    # 磁気赤道面ピッチ角が90度より大きいとき
    if aeq > math.pi/2:
        # 最初南向き...veqは正
        mirlam = mirror(math.pi - aeq)
    else:
        # 最初北向き...veqは負
        mirlam = mirror(aeq)
        veq = -veq

    # トレース開始
    dt2 = dt*0.5

    # 座標配列
    trace = np.zeros((int(tsize/h), 3))
    kk = 0

    # ルンゲクッタ
    # print('RK4 START')
    for k in range(tsize-1):
        F1 = ode(xv, t, veq, aeq, Beq)
        F2 = ode(xv+dt2*F1, t+dt2, veq, aeq, Beq)
        F3 = ode(xv+dt2*F2, t+dt2, veq, aeq, Beq)
        F4 = ode(xv+dt*F3, t+dt, veq, aeq, Beq)
        xv2 = xv + dt*(F1 + 2*F2 + 2*F3 + F4)/6
        # xvの中身... [0]-[2]: gyro中心の座標(trace座標系)

        # 木星中心の座標
        x = xv2[0] + R0x
        y = xv2[1] + R0y
        z = xv2[2] + R0z

        # 北側のミラーポイント
        # ミラーポイントがz_pしきい値よりも下にある場合
        # nakami がほぼゼロになる = ミラー
        # ミラーしたら、z座標を少し下にずらす&veqの符号を反転させる
        # veq < 0 ... 北向き
        r = math.sqrt(x**2 + y**2 + z**2)
        nakami = (x**2 + y**2)**3 - (r**5 * math.sin(aeq)
                                     ** 2 * math.sqrt(r**2 + 3 * z**2))
        if math.sqrt(nakami) < 0.001 and xv2[2] > 0 and veq < 0:
            print('MIRROR NORTH')
            veq = -veq
            xv2[2] += -0.4

        # 南側のミラーポイント
        # ミラーポイントがz_mしきい値よりも上にある場合
        # nakami がほぼゼロになる = ミラー
        # ミラーしたら、z座標を少し上にずらす&veqの符号を反転させる
        # veq > 0 ... 南向き
        if math.sqrt(nakami) < 0.001 and xv2[2] < 0 and veq > 0:
            print('MIRROR SOUTH')
            veq = -veq
            xv2[2] += 0.4

        if k % h == 0:
            # 1ステップでどれくらい進んだか
            # D = np.linalg.norm(xv2-xv)
            # print(D)
            trace[kk, :] = np.array([xv2[0], xv2[1], xv2[2]])
            kk += 1

        # 座標更新
        xv = xv2
        t += dt

        # シミュレーションボックスの外(上)に出たら復帰座標を計算
        if xv[2] > z_p:
            print('UPPER REVERSED')
            xv = comeback(xv, req, z_p_rad, mirlam)
            veq = -veq  # 磁力線に平行な速度成分 向き反転

        # 磁気赤道面に戻ってきたら終了する
        if xv2[2] < - 2:
            print('BREAK')
            break

    return trace[0:kk, :]


#
#
# %% 4次ルンゲクッタ.. classの定義
class RK4:
    def __init__(self, xv, t, dt, tsize, veq, aeq):
        a = rk4(xv, t, dt, tsize, veq, aeq)
        self.positions = a


#
#
# %% トレースを行うfunction
def calc(r0, phiJ0, z0):
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
dt = float(5E-6)  # 時間刻みはEuropaの近くまで来たらもっと細かくして、衝突判定の精度を上げよう
t_len = 1000
# t = np.arange(0, 60, dt)     # np.arange(0, 60, dt)
tsize = int(t_len/dt)


#
#
# %% main関数
def main():
    # 初期座標
    r01 = r_grid.reshape(r_grid.size)  # 1次元化
    phiJ01 = phiJ_grid.reshape(phiJ_grid.size)  # 1次元化
    z01 = z_grid.reshape(z_grid.size)  # 1次元化

    # 初期座標
    x01 = r01[0]*math.cos(phiJ01[0]) - R0x
    y01 = r01[0]*math.sin(phiJ01[0]) - R0y
    z01 = z01[0]
    xv = np.array([x01, y01, z01])

    # 1粒子トレース
    dt = abs(1/(1E+5 + v0eq*math.cos(alphaeq))) * 100
    # print(dt)
    start = time.time()
    result = RK4(xv, t, dt, tsize, v0eq, alphaeq).positions
    print('%.3f seconds' % (time.time()-start))

    # SAVE
    """
    np.savetxt(
        '/Users/shin/Documents/Research/Europa/Codes/gyrocenter/gyrocenter_1/' +
        str(savename), result
    )
    print('DONE')
    """

    return 0


#
#
# %% EXECUTE
if __name__ == '__main__':
    a = main()
