import numpy as np
from ..sl_cosmology import Dang, G, M_Sun, Mpc, c, rhoc, yr
from ..sl_profiles import deVaucouleurs as deV, gnfw
from scipy.interpolate import interp1d, splev, splint, splrep
from scipy.optimize import brentq, leastsq, minimize_scalar
from scipy.integrate import quad


class LensModel:
    """
    LensModel 类：
    所有内部计算单位：
    - M_star, M_halo：Msun（太阳质量）
    - Re, rs, x, R：kpc
    - Sigma, s_cr：Msun / kpc^2
    - alpha(x)：kpc（也提供 arcsec 接口）
    - theta：arcsec（角度单位）
    - kappa, gamma：无量纲
    """
    # use : model = LensModel(logM_star=logM_star,
    #                         logM_halo=logM_halo, logRe=logRe,
    #                         zl=zl, zs=zs, gamma_in=1.0, c_halo=c_halo)

    def __init__(self, logM_star, logM_halo, logRe, zl, zs, gamma_in: float = 1.0, c_halo=5, **kwargs):
        if 'gamma_dm' in kwargs:
            raise ValueError("gamma_dm is no longer supported; use gamma_in.")
        self.zl, self.zs = zl, zs
        self.logM_star = logM_star  # [Msun]
        self.logM_halo = logM_halo  # [Msun]
        self.logRe = logRe            # [kpc]
        self.M_star = 10**logM_star           # [Msun]
        self.M_halo = 10**logM_halo           # [Msun]
        self.Re = 10**logRe                   # [kpc]
        # gNFW inner slope parameter (uses gamma_in everywhere)
        self.gamma_in = float(gamma_in)
        # Halo concentration parameter (dimensionless). Must be provided.
        if c_halo is None:
            raise ValueError("Halo concentration c_halo must be provided.")
        self.c_halo = float(c_halo)
        self._setup_cosmology()               # 计算 s_cr 等
        self._setup_profiles()                # gNFW + deV 分布
        self._setup_splines()                 # 插值缓存

    def _setup_cosmology(self):
        dd = Dang(self.zl)                    # [Mpc]
        ds = Dang(self.zs)                    # [Mpc]
        dds = Dang(self.zs, self.zl)          # [Mpc]
        kpc = Mpc / 1000.                     # [kpc/Mpc]
        self.s_cr = c**2 / (4*np.pi*G) * ds/dds/dd / Mpc / M_Sun * kpc**2  # [Msun / kpc^2]
        self.rhoc_z = rhoc(self.zl)           # [Msun / Mpc^3]

    def _setup_profiles(self):
        r200 = (self.M_halo * 3./(4*np.pi*200*self.rhoc_z))**(1./3.) * 1000.  # [kpc]
        # Compute scale radius from concentration: rs = r200 / c_halo
        if self.c_halo <= 0:
            raise ValueError("Halo concentration c_halo must be positive.")
        self.rs = r200 / self.c_halo            # [kpc]
        # gNFW normalization so that M3d(r200) = M_halo
        self.gnfw_norm = self.M_halo / gnfw.M3d(r200, self.rs, self.gamma_in)

        self.R2d = np.logspace(-3, 2, 1001)     # [R/Re] 无单位
        self.Rkpc = self.R2d * self.rs        # [kpc]
        # Use fast gNFW projected density on a grid
        Sigma = self.gnfw_norm * gnfw.fast_Sigma(self.Rkpc, self.rs, self.gamma_in)  # [Msun / kpc^2]
        self._Sigma = Sigma

    def _setup_splines(self):
        self.sigma_spline = splrep(self.Rkpc, self._Sigma)                # Sigma(R)
        self.sigmaR_spline = splrep(self.Rkpc, self._Sigma * self.Rkpc)  # Sigma(R) * R
        kappa_array = splev(self.Rkpc, self.sigma_spline) / self.s_cr
        self.kappaR_spline = splrep(self.Rkpc, kappa_array * self.Rkpc)  # for gamma

    def alpha_theta(self, theta):  # [arcsec] → [arcsec]
        x = arcsec_to_kpc(theta, self.zl, Dang)        # [kpc]
        alphakpc = self.alpha(x)                       # [kpc]
        return kpc_to_arcsec(alphakpc, self.zl, Dang)  # [arcsec]

    def alpha(self, x):  # [kpc] → [kpc]
        m2d_halo = 2*np.pi * splint(0., abs(x), self.sigmaR_spline)  # [Msun]
        m2d_star = self.M_star * deV.fast_M2d(abs(x)/self.Re)        # [Msun]
        return (m2d_halo + m2d_star) / (np.pi * x * self.s_cr)       # [kpc]
    

    def kappa(self, x):  # [kpc] → [dimensionless]
        sigma_halo = splev(abs(x), self.sigma_spline)                      # [Msun / kpc^2]
        sigma_star = self.M_star * deV.Sigma(abs(x), self.Re)             # [Msun / kpc^2]
        return (sigma_halo + sigma_star) / self.s_cr                      # [dimensionless]
    

    def kappa_star(self, x):  # [kpc] → [dimensionless]
        sigma_star = self.M_star * deV.Sigma(abs(x), self.Re)  # [Msun / kpc^2]
        return sigma_star / self.s_cr


    def gamma(self, x):
        R = abs(x)
        integrand = lambda r: self.kappa(r) * r
        integral, _ = quad(integrand, 0, R, epsabs=1e-8, epsrel=1e-8)
        mean_kappa = 2 * integral / R**2
        return mean_kappa - self.kappa(R)



    def einstein_radius(self):  # [kpc]
        def zerofunc(x): return self.alpha(x) - x
        xmin = 1.01 * self.Rkpc[0]
        xmax = 0.99 * self.Rkpc[-1]
        if zerofunc(xmin)<0:
            return 0.  # 没有爱因斯坦半径
        elif zerofunc(xmax) > 0:
            return np.inf  # 无穷大爱因斯坦半径（无界
        else:
            return brentq(zerofunc, xmin, xmax)
    
    
    def mu_from_rt(self, x):
        """
        使用 μ_r 与 μ_t 计算放大率 (径向/切向放大率法)
        μ_r⁻¹ = 1 + α/x - 2κ
        μ_t⁻¹ = 1 - α/x
        μ = |μ_r · μ_t|
        """
        xabs = abs(x)
        alpha_val = self.alpha(xabs)
        kappa_val = self.kappa(xabs)

        inv_mur = 1. + alpha_val / xabs - 2. * kappa_val
        inv_mut = 1. - alpha_val / xabs

        if inv_mur * inv_mut == 0:
            return np.inf
        return np.abs(1. / (inv_mur * inv_mut))

    def mu_from_kg(self, x):
        """
        使用 κ 与 γ 计算放大率 (广义公式)
        μ = 1 / [ (1 - κ)^2 - γ^2 ]
        """
        kappa_val = self.kappa(x)
        gamma_val = self.gamma(x)
        denom = (1. - kappa_val)**2 - gamma_val**2
        if denom == 0:
            return np.inf
        return np.abs(1. / denom)

    
    def solve_xradcrit(self):  # [kpc]
        """求解径向临界曲线的位置 x_radcrit（图像平面，单位：kpc）"""
        def radial_invmag(x):
            return 1. + self.alpha(x)/x - 2.*self.kappa(x)

        xmin = 1.01 * self.Rkpc[0]
        xmax = 0.99 * self.Rkpc[-1]

        if radial_invmag(xmin) * radial_invmag(xmax) > 0:
            return None  # 没有临界曲线
        return brentq(radial_invmag, xmin, xmax)


    def solve_ycaustic(self):  # [kpc]
        """由 x_radcrit 推出径向 caustic 的 y 值（源平面，单位：kpc）"""
        xradcrit = self.solve_xradcrit()
        if xradcrit is None:
            return 3 * self.einstein_radius()
        return  self.alpha(xradcrit) - xradcrit

    def solve_ycaustic_arcsec(self):
        """返回 ycaustic，单位为 arcsec"""
        y_kpc = self.solve_ycaustic()
        return kpc_to_arcsec(y_kpc, self.zl, Dang)
    
    def Mh5(self):
        """
        返回 halo 在 5 kpc 内的 3D 质量 (单位 Msun)
        """
        R = 5.0  # kpc
        return self.gnfw_norm * gnfw.M3d(R, self.rs, self.gamma_in)
    
    def logMh5(self):
        """
        返回 log10(Mh5)
        """
        return np.log10(self.Mh5())






def arcsec_to_kpc(arcsec, z_l, Dang_func):

    D_d = Dang_func(z_l)  # in Mpc
    arcsec2rad = np.deg2rad(1. / 3600.)  # 1 arcsecond in radians
    kpc = arcsec * arcsec2rad * D_d * 1000.  # kpc conversion (in Mpc)
    return kpc
def kpc_to_arcsec(kpc, z_l, Dang_func):

    D_d = Dang_func(z_l)  # in Mpc
    kpc2rad = kpc / (D_d * 1000.)  # Convert kpc to radians
    arcsec = np.rad2deg(kpc2rad) * 3600.  # Convert radians to arcseconds
    return arcsec
