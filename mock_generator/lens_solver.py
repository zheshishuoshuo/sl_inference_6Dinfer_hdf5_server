from .lens_model import LensModel
from scipy.interpolate import splrep, splint
from ..sl_cosmology import Dang, Mpc, c, G, M_Sun, rhoc
import numpy as np
from ..sl_profiles import nfw, gnfw, deVaucouleurs as deV
from scipy.optimize import brentq


# Default halo concentration when not provided upstream.
# This value is consistent with mass_sampler.generate_samples (c_halo = 5).
DEFAULT_C_HALO = 5.0

def solve_single_lens(model, beta_unit):
   caustic_max_at_lens_plane = model.solve_xradcrit()  # [kpc]
   caustic_max_at_source_plane = model.solve_ycaustic()  # [kpc]
   beta = beta_unit * caustic_max_at_source_plane  # [kpc]
   einstein_radius = model.einstein_radius()  # [kpc]

   def lens_equation(x):
       return model.alpha(x) - x + beta
   
   xA = brentq(lens_equation, einstein_radius, 300*einstein_radius)
   xB = brentq(lens_equation, -einstein_radius, -caustic_max_at_lens_plane)
   # print(caustic_max_at_lens_plane, caustic_max_at_source_plane, beta)
   return xA, xB



# def solve_lens_parameters_from_obs(xA_obs, xB_obs, logRe_obs, logM_halo, zl, zs, gamma_in=1.0, c_halo=5):
#     """
#     从观测像位 (xA_obs, xB_obs) 反推出 logM_star，并计算两像放大率 muA, muB。
#     使用 gNFW + deV，halo 用 M3d 归一化；与正向 LensModel 一致。
#     返回: logM_star_solved, beta_unit, caustic_max, muA, muB
#     """
#     # ---------- 星系/宇宙学 ----------
#     Re = 10**logRe_obs                         # [kpc]
#     M_halo = 10**logM_halo                     # [Msun]
#     dd  = Dang(zl); ds = Dang(zs); dds = Dang(zs, zl)
#     kpc = Mpc / 1000.0
#     s_cr = c**2 / (4*np.pi*G) * ds/dds/dd / Mpc / M_Sun * kpc**2
#     rhoc_z = rhoc(zl)

#     # ---------- gNFW halo ----------
#     r200 = (M_halo * 3.0/(4*np.pi*200*rhoc_z))**(1.0/3.0) * 1000.0  # [kpc]
#     rs   = r200 / c_halo
#     gnfw_norm = M_halo / gnfw.M3d(r200, rs, gamma_in)

#     R2d   = np.logspace(-3, 2, 1001)
#     Rkpc  = R2d * rs
#     Sigma_halo   = gnfw_norm * gnfw.fast_Sigma(Rkpc, rs, gamma_in)
#     sigmaR_spline = splrep(Rkpc, Sigma_halo * Rkpc)

#     # ---------- α 贡献项（注意：剖面里用 |x|，方程分母保留 x 的正负号） ----------
#     def alpha_star_unit(x):
#         return deV.fast_M2d(abs(x)/Re) / (np.pi * x * s_cr)

#     def alpha_halo(x):
#         m2d = 2.0*np.pi * splint(0.0, abs(x), sigmaR_spline)
#         return m2d / (np.pi * x * s_cr)

#     # ---------- 线性消元解 M_star 与 beta ----------
#     denom = alpha_star_unit(xA_obs) - alpha_star_unit(xB_obs)
#     if not np.isfinite(denom) or denom == 0.0:
#         return np.nan, np.nan, np.nan, np.nan, np.nan

#     M_star_solved = ((xA_obs - xB_obs) + alpha_halo(xB_obs) - alpha_halo(xA_obs)) / denom
#     if not np.isfinite(M_star_solved) or M_star_solved <= 0.0:
#         return np.nan, np.nan, np.nan, np.nan, np.nan

#     beta_solved = -(alpha_star_unit(xA_obs)*(xB_obs - alpha_halo(xB_obs))
#                     - alpha_star_unit(xB_obs)*(xA_obs - alpha_halo(xA_obs))) / denom

#     logM_star_solved = np.log10(M_star_solved)

#     # ---------- 组装完整模型：求 caustic、beta_unit、放大率 ----------
#     model = LensModel(
#         logM_star=logM_star_solved,
#         logM_halo=logM_halo,
#         logRe=logRe_obs,
#         zl=zl, zs=zs,
#         gamma_in=gamma_in,
#         c_halo=c_halo,
#     )
#     caustic_max = model.solve_ycaustic()
#     if caustic_max is None or not np.isfinite(caustic_max):
#         return np.nan, np.nan, np.nan, np.nan, np.nan

#     beta_unit = beta_solved / caustic_max
#     if not np.isfinite(beta_unit) or beta_unit < 0 or beta_unit > 1:
#         return np.nan, np.nan, np.nan, np.nan, np.nan

#     # 两像放大率（LensModel 内部实现支持用径向/切向放大率法）
#     muA = model.mu_from_rt(xA_obs)
#     muB = model.mu_from_rt(xB_obs)

#     return logM_star_solved, beta_unit, caustic_max, muA, muB




def solve_lens_parameters_from_obs(xA_obs, xB_obs, logRe_obs, logM_halo, zl, zs, gamma_in=1.0, c_halo=5):
    """
    用 gNFW 替代 NFW，并加入 gamma_in（暗物质内斜率）
    """

    # ---------- 星系参数 ----------
    Re = 10**logRe_obs  # [kpc]
    M_halo = 10**logM_halo  # [Msun]

    # ---------- 计算 cosmology ----------
    dd = Dang(zl)  # [Mpc]
    ds = Dang(zs)  # [Mpc]
    dds = Dang(zs, zl)  # [Mpc]
    kpc = Mpc / 1000.  # [kpc/Mpc]
    s_cr = c**2 / (4*np.pi*G) * ds/dds/dd / Mpc / M_Sun * kpc**2  # [Msun/kpc^2]
    rhoc_z = rhoc(zl)

    # ---------- gNFW halo ----------
    r200 = (M_halo * 3./(4*np.pi*200*rhoc_z))**(1./3.) * 1000.  # [kpc]
    rs = r200 / c_halo  # scale radius
    gnfw_norm = M_halo / gnfw.M3d(r200, rs, gamma_in)

    R2d = np.logspace(-3, 2, 1001)  # [R/Re] 无单位
    Rkpc = R2d * rs  # [kpc]
    Sigma_halo = gnfw_norm * gnfw.fast_Sigma(Rkpc, rs, gamma_in)
    sigmaR_spline = splrep(Rkpc, Sigma_halo * Rkpc)

    # ---------- alpha ----------
    def alpha_star_unit(x):
        return deV.fast_M2d(abs(x)/Re) / (np.pi * x * s_cr)

    def alpha_halo(x):
        m2d = 2*np.pi * splint(0., abs(x), sigmaR_spline)
        return m2d / (np.pi * x * s_cr)

    # ---------- 解 M_star ----------
    M_star_solved = ((xA_obs - xB_obs) + alpha_halo(xB_obs) - alpha_halo(xA_obs)) / \
                    (alpha_star_unit(xA_obs) - alpha_star_unit(xB_obs))

    beta_solved = -(alpha_star_unit(xA_obs)*(xB_obs-alpha_halo(xB_obs)) -
                    alpha_star_unit(xB_obs)*(xA_obs-alpha_halo(xA_obs))) / \
                    (alpha_star_unit(xB_obs) - alpha_star_unit(xA_obs))
    
    if M_star_solved <= 0.0 or beta_solved < 0.0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    logM_star_solved = np.log10(M_star_solved)

    # try:
    model = LensModel(
        logM_star=logM_star_solved,
        logM_halo=logM_halo,
        logRe=logRe_obs,
        zl=zl, zs=zs,
        gamma_in=gamma_in,  # ✅ 加入 gamma_in
        c_halo=c_halo
    )
    caustic_max = model.solve_ycaustic()
    if caustic_max is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    beta_unit = beta_solved / caustic_max
    # except:

    #     return np.nan, np.nan, np.nan

    if beta_unit < 0 or beta_unit > 1:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    muA = model.mu_from_rt(xA_obs)
    muB = model.mu_from_rt(xB_obs)

    return logM_star_solved, beta_unit, caustic_max, muA, muB






def compute_detJ(theta1_obs, theta2_obs, logRe_obs, logMh, zl=0.3, zs=2.0, gamma_in: float = 1.0):
    # 相对步长（按像位尺度自适应）
    h1 = 1e-4 * max(1.0, abs(theta1_obs))
    h2 = 1e-4 * max(1.0, abs(theta2_obs))

    def solve(theta1, theta2):
        return solve_lens_parameters_from_obs(theta1, theta2, logRe_obs, logMh, zl, zs, gamma_in=gamma_in)

    # 中心差分
    logM_p, beta_p, yc_p, _, _ = solve(theta1_obs + h1, theta2_obs)
    logM_m, beta_m, yc_m, _, _ = solve(theta1_obs - h1, theta2_obs)
    dlogM_dtheta1 = (logM_p - logM_m) / (2*h1)
    dbeta_dtheta1 = ((beta_p*yc_p) - (beta_m*yc_m)) / (2*h1)

    logM_p, beta_p, yc_p, _, _ = solve(theta1_obs, theta2_obs + h2)
    logM_m, beta_m, yc_m, _, _ = solve(theta1_obs, theta2_obs - h2)
    dlogM_dtheta2 = (logM_p - logM_m) / (2*h2)
    dbeta_dtheta2 = ((beta_p*yc_p) - (beta_m*yc_m)) / (2*h2)

    J = np.array([[dlogM_dtheta1, dlogM_dtheta2],
                  [dbeta_dtheta1,  dbeta_dtheta2]])
    return np.abs(np.linalg.det(J))








###

    # import sys
    # sys.path.append("../../")  # 根据你的 notebook 路径设置

    # from sl_inference_only_muDMalpha.mock_generator.lens_model import LensModel
    # from sl_inference_only_muDMalpha.mock_generator.lens_solver import solve_single_lens, solve_lens_parameters_from_obs_yc
    # model = LensModel(logM_star=11.5, logM_halo=13.0, logRe=0.8, zl=0.3, zs=2.0)
    # xA, xB = solve_single_lens(model,0.5)
    # Mstar, beta,yc = solve_lens_parameters_from_obs_yc(xA, xB, 0.8, 13, 0.3, 2)
    # print(Mstar, beta, yc)
