"""Full 2D-kernel likelihood for 6D hyper-parameters.

Consumes LensGrid2D tables and performs the marginal likelihood integral over
(logMh, gamma_h) per lens, then aggregates across lenses and includes the
selection term A(η).

η ordering (6D): (alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma)
"""

from __future__ import annotations

from typing import Iterable, Sequence, Optional, Callable

# No internal multiprocessing pools here; outer layers may parallelize walkers

import os
import numpy as np
from scipy.stats import norm, skewnorm
from scipy.special import logsumexp

from .cached_A import cached_A_interp
from .make_tabulate import LensGrid2D, tabulate_likelihood_grids
from .mock_generator.mass_sampler import MODEL_PARAMS
from .config import SCATTER
from .utils import selection_function
from .compute_A_eta import load_A_eta_interpolator




MODEL_P = MODEL_PARAMS["deVauc"]




A_INTERP: Optional[Callable[[Sequence[float]], float]] = None
# # Accept either a 6D table (preferred) or fall back to existing 5D table via adapter
# file_path = os.path.join(os.path.dirname(__file__), 'aeta_tables', 'Aeta6D_mu50_beta50_sigma50_mugamma50_sigmagamma50_alpha50.h5')
# A_INTERP = load_A_eta_interpolator(file_path)

def init_a_interpolator():
    """确保插值器仅在主进程加载一次，可被子进程共享。"""
    global A_INTERP
    if A_INTERP is None:
        file_path = os.path.join(
            os.path.dirname(__file__),
            'aeta_tables',
            'Aeta6D_mu30_beta30_sigma30_mugamma30_sigmagamma30_alpha30.h5',
        )
        A_INTERP = load_A_eta_interpolator(file_path)
    return A_INTERP

def A_interp(eta: Sequence[float]) -> float:
    """Evaluate A(η) interpolator at given η.

    η ordering (6D): (alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma)
    """
    if A_INTERP is None:
        raise RuntimeError("A(eta) interpolator is not initialized.")

    alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma = eta
    # Interpolator expects grid ordering (mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma, alpha)
    eta6 = (mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma, alpha_sps)

    return A_INTERP(eta6)

# ----------------------------------------------------------------------------
# Helpers for numerical safety
# ----------------------------------------------------------------------------
def safe_value(v: float, *, minval: float = 1e-300) -> float:
    """Return a finite scalar with a lower bound.

    - Converts NaN/inf to finite values via np.nan_to_num
    - Clamps to at least ``minval`` to protect against log(0)
    """
    try:
        vv = float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
    except Exception:
        vv = 0.0
    if not np.isfinite(vv):
        vv = 0.0
    if minval is not None:
        vv = max(vv, minval)
    return vv


# Removed K(mu1, mu2) table dependency; photometric selection and likelihood
# are handled by explicit integration over source magnitude ms per lens.

# Convenience wrapper ---------------------------------------------------------


# Priors ---------------------------------------------------------------------



def log_prior(theta: Sequence[float]) -> float:
    """Flat/top-hat prior for 6D hyperparameters η,
    matched to the A(η) grid support.
    η = (alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma)
    """
    if len(theta) != 6:
        return -np.inf

    alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma = map(float, theta)

    # --- 必须严格限制在 A(η) 网格范围内 ---
    if not (-0.29 <= alpha_sps <= 0.29):
        return -np.inf
    if not (12.1 <= mu_h <= 13.9):
        return -np.inf
    if not (1.1 <= beta_h <= 2.9):
        return -np.inf
    if not (0.1 <= sigma_h <= 0.55):
        return -np.inf
    if not (0.51 <= mu_gamma <= 1.49):
        return -np.inf
    if not (0.16 <= sigma_gamma <= 0.24):
        return -np.inf

    return 0.0  # 平坦先验



    # mu_DM_grid = np.linspace(12, 14, N)
    # beta_DM_grid = np.linspace(1, 3, 15)
    # sigma_DM_grid = np.linspace(0.01, 0.6, 15)
    # mu_gamma_grid = np.linspace(0.5, 1.5, N)
    # sigma_gamma_grid = np.linspace(0.01, 0.3, N)
    # alpha_grid = np.linspace(-0.5, 0.5, N)

# def log_prior(theta: Sequence[float]) -> float:
#     """Flat/top-hat prior for 6D parameters η.

#     η = (alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma)
#     """
#     if len(theta) != 6:
#         return -np.inf
#     alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma = map(float, theta)

#     # Broad, physically sensible ranges. Adjust in future if needed.
#     # if not (0.0 < alpha_sps < 0.3):
#     #     return -np.inf
#     # if not (11.0 < mu_h < 16.0):
#     #     return -np.inf
#     # if not (1.0 < beta_h < 5.0):
#     #     return -np.inf
#     # if not (0.2 < sigma_h < 0.6):
#     #     return -np.inf
#     # if not (0.5 < mu_gamma < 4):
#     #     return -np.inf
#     # if not (0.05 < sigma_gamma < 2):
#     #     return -np.inf
#     return 0.0

    # mu_DM_grid = np.linspace(12.6, 13.2, 30)
    # sigma_DM_grid = np.linspace(0.27, 0.47, 10)
    # alpha_grid = np.linspace(0.1, 0.2, 50)


# Single-lens integral -------------------------------------------------------


'''
likelihood terms

1. msps noise term: P(m_sps_obs | m_sps_true, sigma_star)
2. msps prior term: P(m_sps_true) (skew-normal)
3. size-mass term: P(logRe_obs | m_sps_true)
4. halo-stellar prior: P(logMh | m_sps_true, mu_h, beta_h, sigma_h)
5. photometric selection + likelihood term:
   factor_constant = beta * |detJ| * K_phot
6. hyperprior on gamma_h: P(gamma_h | mu_gamma, sigma_gamma)



'''

SPS_A, SPS_LOC, SPS_SCALE = (float(10 ** MODEL_P["log_s_star"]), float(MODEL_P["mu_star"]), float(MODEL_P["sigma_star"]))

def single_lens_likelihood(
    grid: LensGrid2D,
    eta: Sequence[float],
) -> float:
    """Per-lens marginal likelihood B(d_i | η) using the 2D kernel.

    B(d_i | η) = ∬ F_i(logMh, gamma_h) P(logMh | η) P(gamma_h | η) dlogMh dgamma_h

    F_i combines observational constraints and explicit photometric
    selection via integration over source magnitude ms:

    - Observational terms: SPS-mass likelihood and size–mass relation.
    - Population terms: skew-normal prior on SPS mass and halo–stellar scaling.
    - Photometric + selection term: ∫ L1 L2 p_det(μ1) p_det(μ2) p(ms) dms.
    """

    if len(eta) != 6:
        return 1e-300
    alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma = map(float, eta)


    # Axes from grid
    M_halo_axis = grid.logMh_axis
    Gamma_h_axis = grid.gamma_h_axis
    
    # Vakues from grid
    logM_star_true_grid = grid.logM_star_true
    logM_sps_obs = grid.logM_star_sps_obs
    logRe_obs = grid.logRe
    sigma_star = SCATTER.star
    Msps_grid = logM_star_true_grid - alpha_sps




    # weight  


    # 1. msps noise term: P(m_sps_obs | m_sps_true, sigma_star)
    # 2. msps prior term: P(m_sps_true) (skew-normal)
    # 3. size-mass term: P(logRe_obs | m_sps_true)
    # 4. halo-stellar prior: P(logMh | m_sps_true, mu_h, beta_h, sigma_h)
    # 5. factor_constant = beta * |detJ| * K_phot
    # 6. hyperprior on gamma_h: P(gamma_h | mu_gamma, sigma_gamma)


    P_Msps_obs = norm.pdf(logM_sps_obs, loc=Msps_grid, scale=sigma_star)
    P_Msps_prior = skewnorm.pdf(Msps_grid, a=SPS_A, loc=SPS_LOC, scale=SPS_SCALE)
    P_logRe = norm.pdf(logRe_obs, loc=(MODEL_P["mu_R0"] + MODEL_P["beta_R"] * (Msps_grid - 11.4)), scale=MODEL_P["sigma_R"])
    P_logMh_grid = norm.pdf(M_halo_axis[:, None], loc=(mu_h + beta_h * (Msps_grid - 11.4)), scale=max(float(sigma_h), 1e-12))
    factors_constant_grid = grid.factors_constant
    # Truncated-normal prior for gamma on [0.4, 1.6]
    _g_lo, _g_hi = 0.4, 1.6
    _Zg = float(norm.cdf(_g_hi, loc=mu_gamma, scale=sigma_gamma) - norm.cdf(_g_lo, loc=mu_gamma, scale=sigma_gamma))
    if not np.isfinite(_Zg) or _Zg <= 0:
        _Zg = 1e-12
    P_gamma_h_grid = norm.pdf(Gamma_h_axis[None, :], loc=mu_gamma, scale=sigma_gamma) / _Zg



    integrand = P_Msps_obs * P_Msps_prior * P_logRe * P_logMh_grid * factors_constant_grid * P_gamma_h_grid

    # nan to zero
    integrand = np.nan_to_num(integrand)

    I_gamma = np.trapz(integrand, x=Gamma_h_axis, axis=1)
    likelihood = np.trapz(I_gamma, x=M_halo_axis, axis=0)

    likelihood = np.nan_to_num(likelihood, nan=0.0, posinf=0.0, neginf=0.0)
    

    return likelihood




# Public API -----------------------------------------------------------------


def _worker_wrapper(args):
    g, eta = args
    return single_lens_likelihood(g, eta)
    
def log_likelihood(
    eta: Sequence[float],
    grids: Sequence[LensGrid2D],
    *,
    pool: Optional[object] = None,
) -> float:
    """Joint log-likelihood for all lenses using 2D kernels (6D η).

    logL(η) = ∑_i log B(d_i | η) − N_lens log A(η)
    """
    A_eta = A_interp(list(eta))
    if np.isnan(A_eta) or A_eta <= 0.0:
        # print(eta)
        raise ValueError("A(eta) is non-positive or NaN."+str(eta)+str(A_eta))
    # A_eta = 1



    if pool is not None and hasattr(pool, "map"):
        args_list = [(g, eta) for g in grids]  # ✅ 显式打包 eta
        results = list(pool.map(_worker_wrapper, args_list))
    else:
        results = [single_lens_likelihood(g, eta) for g in grids]


    results = np.asarray(results, dtype=float)
    logLs = np.log(results)
    total = np.sum(logLs) - len(grids) * np.log(A_eta)

    return total



def log_posterior(
    eta: Sequence[float],
    grids: Sequence[LensGrid2D],
    *,
    pool: Optional[object] = None
) -> float:
    """Posterior = prior + likelihood for 5D hyper-parameters."""

    # eta[4] = 1.0
    # eta[5] = 0.2

    lp = log_prior(eta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(eta, grids, pool=pool)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


__all__ = [
    "precompute_grids",
    "log_prior",
    "log_likelihood",
    "log_posterior",
]


# test
if __name__ == "__main__":
    import emcee
    import multiprocessing as mp
    from pathlib import Path
    import pandas as pd
    
    from .main import DMGrid2D

    # Example MCMC run using the 2D-kernel likelihood

    # --- 载入预计算的 LensGrid2D 表格 ---
    from .make_tabulate import tabulate_likelihood_grids

    etaalpha = np.array([[alpha, 12.91, 2.04, 0.37, 1.0, 0.2] for alpha in np.linspace(0.05, 0.2, 40)])

    etadm = np.array([[0.1, mu, 2.04, 0.37, 1.0, 0.2] for mu in np.linspace(12.6, 13.2, 40)])

    etagamma = np.array([[0.1, 12.91, 2.04, 0.37, mu_gamma, 0.2] for mu_gamma in np.linspace(0.5, 1.5, 40)])

    dm_grid = DMGrid2D(
        logMh=np.linspace(12.0, 13.5, 10),
        gamma_h=np.linspace(0.5, 1.5, 10),
    )

    _mock_observed_data = pd.DataFrame({
        "xA": [5.644504],
        "xB": [-3.957179],
        "logM_star_sps_observed": [11.400591],
        "logRe": [0.985400],
        "magnitude_observedA": [25.403552],
        "magnitude_observedB": [25.591093],
    })
    grids = tabulate_likelihood_grids(mock_observed_data=_mock_observed_data, dm_grid=dm_grid)
    post = []
    
    e = "gamma"

    if e == "alpha":
        eta = etaalpha
        x = etaalpha[:, 0]
        truth = 0.1
    elif e == "dm":
        eta = etadm
        x = etadm[:, 1]
        truth = 12.91
    elif e == "gamma":
        eta = etagamma
        x = etagamma[:, 4]
        truth = 1

    with mp.Pool(processes=4) as pool:
        for e in eta:
            p = log_posterior(e, grids, pool=pool)
            post.append(p)
    # print(post)
    import matplotlib.pyplot as plt
    plt.plot(x, post, marker='o')
    plt.axvline(truth, color='r', linestyle='--', label='Truth')
    plt.xlabel(e)
    plt.ylabel("log posterior")
    plt.show()
