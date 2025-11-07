"""2D kernel tabulation over (logMh, gamma_h) for each observed lens.

Builds, per observed lens and per dark-matter grid point (logMh, gamma_h),
the required stellar mass logM_star_true and image magnifications (muA, muB)
by strictly calling the internal physics in mock_generator:

- Uses solve_lens_parameters_from_obs for parameter inversion and magnifications
- Relies on LensModel internally (no custom lensing math reimplementation)
- Populates NaN on failure per grid point without aborting a lens

The public API is unchanged: tabulate_likelihood_grids returns a list of
LensGrid2D objects compatible with the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, as_completed

from .mock_generator.lens_model import LensModel  # used by solver internally
from .mock_generator.lens_solver import solve_lens_parameters_from_obs
from .mock_generator.lens_solver import compute_detJ

from .utils import selection_function

# Optional progress bar
try:  # tqdm is already used elsewhere in the project
    from tqdm import tqdm  # type: ignore
except Exception:  # Fallback noop if tqdm is unavailable
    def tqdm(x, **kwargs):  # type: ignore
        return x

DEFAULT_C_HALO = 5.0


ALPHA_S = -1.3
M_S_STAR = 24.5
MS_MIN, MS_MAX = 20.0, 30.0
MS_GRID = np.linspace(MS_MIN, MS_MAX, 1000)


def _source_mag_prior(ms: np.ndarray) -> np.ndarray:
    L = 10 ** (-0.4 * (ms - M_S_STAR))
    return L ** (ALPHA_S + 1) * np.exp(-L)


P_MS = _source_mag_prior(MS_GRID)
P_MS /= np.trapz(P_MS, MS_GRID)


def photometric_factor(muA, muB, m1_obs, m2_obs, m_lim, sigma_m,
                       ms_grid, p_ms) -> float:
    """
    计算 ∫ L1(ms) L2(ms) p_det(muA,ms) p_det(muB,ms) p(ms) dms
    使用 numpy.trapz 进行梯形积分。
    """
    # ---- 1. 观测光度似然 ----
    m1_model = ms_grid - 2.5 * np.log10(np.clip(muA, 1e-6, None))
    m2_model = ms_grid - 2.5 * np.log10(np.clip(muB, 1e-6, None))
    L1 = norm.pdf(m1_obs, loc=m1_model, scale=sigma_m)
    L2 = norm.pdf(m2_obs, loc=m2_model, scale=sigma_m)

    # ---- 2. 选择函数 ----
    P1 = selection_function(np.array([muA]), m_lim, ms_grid[None, :], sigma_m)[0]
    P2 = selection_function(np.array([muB]), m_lim, ms_grid[None, :], sigma_m)[0]

    # ---- 3. 被积函数 ----
    integrand = L1 * L2 * P1 * P2 * p_ms

    # ---- 4. 梯形积分 ----
    integral = np.trapz(integrand, x=ms_grid)

    return float(integral)



    # logMh_grid: np.ndarray
    # logM_star: np.ndarray
    # sample_factor: np.ndarray
    # logRe: float
    # # Optional extras for debugging/exports
    # detJ: np.ndarray | None = None
    # beta_unit: np.ndarray | None = None
    # ycaustic: np.ndarray | None = None


    # beta = beta_unit * yc

    # L1 = mag_likelihood(m1_obs, mu1, MS_GRID, sigma_m)
    # L2 = mag_likelihood(m2_obs, mu2, MS_GRID, sigma_m)
    # L3 = selection_function(mu1, m_lim=26.5, ms=MS_GRID, sigma_m=sigma_m)
    # L4 = selection_function(mu2, m_lim=26.5, ms=MS_GRID, sigma_m=sigma_m)
    # Lphot = np.trapz(P_MS * L1 * L2 * L3 * L4, MS_GRID)
    # del L1, L2, L3, L4

    # valid = np.isfinite(mu2) & np.isfinite(mu_rB) & (mu_rB > 0)
    # if not valid:
    #     Lphot = 0.0
    #     detJ = 0.0
    #     beta_unit = 0.0
    #     yc = 0.0

    # sample_factor = beta * abs(detJ) * Lphot




@dataclass
class LensGrid2D:
    """
    Per-lens likelihood kernel over a 2D dark matter grid (logMh, gamma_h).
    Stores only quantities that are independent of hyperparameters η.
    """

    # ---------- Grid axes ----------
    logMh_axis: np.ndarray          # shape = (n_Mh,)
    gamma_h_axis: np.ndarray        # shape = (n_gamma,)

    # ---------- Tabulated physical kernel ----------
    # These are evaluated at every (logMh, gamma_h) grid point
    logM_star_true: np.ndarray      # shape = (n_Mh, n_gamma)
    muA: np.ndarray                 # shape = (n_Mh, n_gamma)
    muB: np.ndarray                 # shape = (n_Mh, n_gamma)
    factors_constant: np.ndarray  # = beta * |detJ| * K_phot  # shape = (n_Mh, n_gamma)

    # ---------- Per-lens observational inputs ----------
    logM_star_sps_obs: float        # observed SPS stellar mass
    xA_obs: float                 # observed image A position
    xB_obs: float                 # observed image B position
    m1_obs: float                   # observed image A magnitude
    m2_obs: float                   # observed image B magnitude
    zl: float                       # lens redshift
    zs: float                       # source redshift

    # ---------- Galaxy structural info ----------
    logRe: float                    # effective radius of lens galaxy

    # ---------- Optional metadata (no math meaning) ----------
    lens_id: int = None
    info: dict = None
    beta_unit: np.ndarray = None
    ycaustic: np.ndarray = None
    L_selection: np.ndarray = None
    L_ms: np.ndarray = None
    beta: np.ndarray = None                # shape = (n_Mh, n_gamma)
    detJ: np.ndarray = None              # shape = (n_Mh, n_gamma)
    K_phot: np.ndarray = None




# 把不需要超参数的部分都提前算好

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def compute_single_lens_grid(
    idx: int,
    xA: float,
    xB: float,
    logRe: float,
    msps_obs: float,
    m1_obs: float,
    m2_obs: float,
    logMh_axis: np.ndarray,
    gamma_axis: np.ndarray,
    zl: float,
    zs: float,
    c_halo: float,
) -> Tuple[int, LensGrid2D]:
    """Worker to compute a full 2D grid for one lens.

    Each grid cell calls solve_lens_parameters_from_obs and records
    logM_star_true, muA, muB. Failures fill NaN and do not abort.
    """
    nMh = int(logMh_axis.size)
    nG = int(gamma_axis.size)
    logM_star_true = np.full((nMh, nG), np.nan, dtype=float)

    muA = np.full((nMh, nG), np.nan, dtype=float)
    muB = np.full((nMh, nG), np.nan, dtype=float)
    beta_unit = np.full((nMh, nG), np.nan, dtype=float)
    ycaustic = np.full((nMh, nG), np.nan, dtype=float)
    detJ = np.full((nMh, nG), np.nan, dtype=float)
    K_phot = np.full((nMh, nG), np.nan, dtype=float)


    # Nested loops over the DM grid
    for i in range(nMh):
        logMh = float(logMh_axis[i])
        for j in range(nG):
            gamma_in = float(gamma_axis[j])

            logM_star_true_ij, beta_u_ij, yc_ij, muA_ij, muB_ij = solve_lens_parameters_from_obs(
                xA_obs=xA,
                xB_obs=xB,
                logRe_obs=logRe,
                logM_halo=logMh,
                zl=zl,
                zs=zs,
                gamma_in=gamma_in,
                c_halo=c_halo,
            )

            
            logM_star_true[i, j] = logM_star_true_ij
            muA[i, j] = muA_ij
            muB[i, j] = muB_ij
            beta_unit[i, j] = beta_u_ij
            ycaustic[i, j] = yc_ij

            detJ[i, j] = compute_detJ(xA, xB, logRe, logMh, zl=zl, zs=zs, gamma_in=gamma_in)


            K_phot[i, j] = photometric_factor(
                muA=muA_ij,
                muB=muB_ij,
                m1_obs=m1_obs,
                m2_obs=m2_obs,
                m_lim=26.5,
                sigma_m=0.1,
                ms_grid=MS_GRID,
                p_ms=P_MS,
            )

    # 计算 beta * |detJ| * K_phot
    factors_constant = beta_unit * ycaustic * np.abs(detJ) * K_phot


    grid = LensGrid2D(
        logMh_axis=np.asarray(logMh_axis, dtype=float),
        gamma_h_axis=np.asarray(gamma_axis, dtype=float),
        logM_star_true=logM_star_true,
        muA=muA,
        muB=muB,
        logM_star_sps_obs=float(msps_obs),
        xA_obs=float(xA),
        xB_obs=float(xB),
        m1_obs=float(m1_obs),
        m2_obs=float(m2_obs),
        zl=float(zl),
        zs=float(zs),
        logRe=float(logRe),
        lens_id=int(idx),
        factors_constant=factors_constant
    )

    return idx, grid




def prepare_lens_tasks(mock_observed_data: pd.DataFrame) -> List[Tuple]:
    """Prepare per-lens computation tasks from observed data."""
    tasks = []
    for idx, row in enumerate(mock_observed_data.itertuples(index=False)):
        try:
            xA = float(getattr(row, "xA"))
            xB = float(getattr(row, "xB"))
            logRe_val = float(getattr(row, "logRe"))
            msps_obs = float(getattr(row, "logM_star_sps_observed"))
            m1 = float(getattr(row, "magnitude_observedA"))
            m2 = float(getattr(row, "magnitude_observedB"))
            ch = DEFAULT_C_HALO
        except Exception:
            # Missing or invalid data → fill with NaN but keep alignment
            xA = xB = logRe_val = msps_obs = m1 = m2 = np.nan
            ch = DEFAULT_C_HALO
        tasks.append((idx, xA, xB, logRe_val, msps_obs, m1, m2, ch))
    return tasks


def run_serial_lens_grids(
    tasks: List[Tuple],
    logMh_axis: np.ndarray,
    gamma_axis: np.ndarray,
    zl: float,
    zs: float,
    show_progress: bool = True,
) -> List["LensGrid2D"]:
    """Compute all lens grids sequentially."""
    results: List["LensGrid2D"] = []
    iterator = tqdm(tasks, total=len(tasks), desc="Lenses", dynamic_ncols=True) if show_progress else tasks

    for t in iterator:
        idx, xA, xB, logRe_val, msps_obs, m1, m2, ch = t
        _, grid = compute_single_lens_grid(
            idx, xA, xB, logRe_val, msps_obs, m1, m2,
            logMh_axis, gamma_axis, zl, zs, ch
        )
        results.append(grid)
    return results


def run_parallel_lens_grids(
    tasks: List[Tuple],
    logMh_axis: np.ndarray,
    gamma_axis: np.ndarray,
    zl: float,
    zs: float,
    n_jobs: int,
    show_progress: bool = True,
) -> List["LensGrid2D"]:
    """Compute all lens grids in parallel using ProcessPoolExecutor."""
    n_lenses = len(tasks)
    ordered: List[Optional["LensGrid2D"]] = [None] * n_lenses
    pbar = tqdm(total=n_lenses, desc="Lenses", dynamic_ncols=True) if show_progress else None

    def job_wrap(tup):
        idx, xA, xB, logRe_val, msps_obs, m1, m2, ch = tup
        return compute_single_lens_grid(
            idx, xA, xB, logRe_val, msps_obs, m1, m2,
            logMh_axis, gamma_axis, zl, zs, ch
        )

    with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
        futures = {ex.submit(job_wrap, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            try:
                idx, grid = fut.result()
                if 0 <= idx < n_lenses:
                    ordered[idx] = grid
            except Exception:
                pass
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()

    # Ensure all results exist
    assert all(g is not None for g in ordered), "Some lens grids failed to compute."
    return ordered


def tabulate_likelihood_grids(
    mock_observed_data: pd.DataFrame,
    dm_grid,
    zl: float = 0.3,
    zs: float = 2.0,
    *,
    show_progress: bool = True,
    n_jobs: Optional[int] = None,
) -> List["LensGrid2D"]:
    """Compute 2D hyperparameter-free kernels for each lens.

    Parameters
    ----------
    mock_observed_data:
        DataFrame with columns: ``xA``, ``xB``, ``logRe``,
        ``magnitude_observedA``, ``magnitude_observedB``,
        ``logM_star_sps_observed``.
    dm_grid:
        An external 2D grid with attributes ``logMh`` and ``gamma_h``.
    zl, zs:
        Lens and source redshifts.

    Returns
    -------
    list[LensGrid2D]
        Per-lens kernel tables over (logMh, gamma_h).
    """

    logMh_axis = np.asarray(dm_grid.logMh, dtype=float)
    gamma_axis = np.asarray(dm_grid.gamma_h, dtype=float)
    tasks = prepare_lens_tasks(mock_observed_data)

    if not n_jobs or int(n_jobs) == 1:
        results = run_serial_lens_grids(tasks, logMh_axis, gamma_axis, zl, zs, show_progress)
    else:
        results = run_parallel_lens_grids(tasks, logMh_axis, gamma_axis, zl, zs, int(n_jobs), show_progress)

    return results



__all__ = ["LensGrid2D", "tabulate_likelihood_grids"]
