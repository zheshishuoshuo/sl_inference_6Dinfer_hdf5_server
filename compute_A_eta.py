"""Monte Carlo evaluation of the normalisation factor A(eta), 6D version.

This module implements the algorithm described in the project notes for
estimating the selection-function normalisation that appears in the
likelihood.  The computation proceeds by Monte Carlo sampling of the lens
population and evaluating the integrand

    T = T1 * T2 * T3

for each sample.  Here:

* ``T1`` is the integral over source magnitude of the detection probability
  for the two lensed images weighted by the source magnitude prior.
* ``T2`` is the weighting from the random source position, proportional to
  the square of the caustic scale ``betamax`` times the uniform variate ``u``.
* ``T3`` is the (untruncated) halo–mass relation ``p(Mh | muDM, Msps)``
  evaluated at the sampled halo mass.

The final estimate of ``A`` is the average of ``T`` over all Monte Carlo
samples with an additional factor from importance sampling the halo mass with
an uninformative uniform proposal.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
from itertools import repeat

import numpy as np
import h5py
from scipy.stats import norm
from tqdm import tqdm

from .config import SCATTER
from .mock_generator.lens_model import LensModel
from .mock_generator.lens_solver import solve_single_lens, solve_lens_parameters_from_obs
from .mock_generator.mass_sampler import MODEL_PARAMS, generate_samples
from .utils import selection_function
from .build_k_table import load_K_interpolator

# -----------------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------------

# def load_K_interpolator(
#     path: str,
#     *,
#     method: Literal["linear", "nearest"] = "linear",
#     bounds_error: bool = False,
#     fill_value: float | None = 0.0,
#     return_arrays: bool = False,
# ):

# Preload K(mu1, mu2) interpolator (integrated over source magnitude with prior)
K_interp = load_K_interpolator(
    os.path.join(os.path.dirname(__file__), "K_K_table_mu1000_ms2000.h5"),
    method="linear",
    bounds_error=False,
    fill_value=0.0,
)

def sample_lens_population(n_samples: int, zl: float = 0.3, zs: float = 2.0):
    """Draw Monte Carlo samples of the lens population.

    Parameters
    ----------
    n_samples
        Number of Monte Carlo samples to draw.
    zl, zs
        Lens and source redshifts.

    Returns
    -------
    dict
        Dictionary containing sampled stellar masses, sizes, halo masses and
        source-position variables along with the bounds of the halo-mass
        proposal distribution.
    """

    data = generate_samples(n_samples)
    logM_star_sps = data["logM_star_sps"]
    logRe = data["logRe"]
    gamma_in = np.random.rand(n_samples) * 2.0  # uniform in [0, 2]
    c_halo = data["c_halo"]
    
    beta_unit = np.random.rand(n_samples)**0.5
    # Uniform proposal for halo mass to allow importance reweighting
    logMh_min, logMh_max = 11.0, 15.0
    logMh = np.random.uniform(logMh_min, logMh_max, n_samples)

    return {
        "logM_star_sps": logM_star_sps,
        "logRe": logRe,
        "beta": beta_unit,
        "gamma_in": gamma_in,
        "logMh": logMh,
        "logMh_min": logMh_min,
        "logMh_max": logMh_max,
        "zl": zl,
        "zs": zs,
    }


# -----------------------------------------------------------------------------
# Lens-equation solver
# -----------------------------------------------------------------------------


def solve_magnification(args):
    """Solve a single lens configuration returning magnifications and caustic."""

    logM_star, logRe, logMh, gamma_in, beta_unit, zl, zs = args
    try:
        model = LensModel(
            logM_star=logM_star, logM_halo=logMh, logRe=logRe, zl=zl, zs=zs, gamma_in=gamma_in,
        )
        xA, xB = solve_single_lens(model, beta_unit)
        ycaustic = model.solve_ycaustic() or 0.0
        mu1 = model.mu_from_rt(xA)
        mu2 = model.mu_from_rt(xB)
        if not (np.isfinite(mu1) and np.isfinite(mu2) and ycaustic > 0):
            return (np.nan, np.nan, 0.0)
        return (mu1, mu2, ycaustic)
    except Exception:
        return (np.nan, np.nan, 0.0)


def compute_magnifications(
    logM_star: np.ndarray,
    logRe: np.ndarray,
    logMh: np.ndarray,
    gamma_in: np.ndarray,
    beta_unit: np.ndarray,
    zl: float,
    zs: float,
    n_jobs: int | None = None,
):
    """Compute magnifications for each Monte Carlo sample."""

    if n_jobs is None:
        # Use half the available CPUs by default
        n_jobs = max(1, (cpu_count() or 2) // 2)

    n = len(logM_star)
    args = zip(logM_star, logRe, logMh, gamma_in, beta_unit, repeat(zl), repeat(zs))
    # Use threads to preserve shared memory and keep tqdm progress
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        results = list(
            tqdm(
                pool.map(solve_magnification, args),
                total=n,
                desc="solving lenses",
                leave=False,
            )
        )
    mu1, mu2, betamax = map(np.array, zip(*results))
    mu1, mu2, betamax = np.nan_to_num(mu1, nan=np.nan), np.nan_to_num(mu2, nan=np.nan), np.nan_to_num(betamax, nan=0.0)
    return mu1, mu2, betamax


# -----------------------------------------------------------------------------
# Source magnitude prior
# -----------------------------------------------------------------------------


def ms_distribution(ms_grid: np.ndarray, alpha_s: float = -1.3, ms_star: float = 24.5):
    """Normalised PDF of the unlensed source magnitude."""

    L = 10 ** (-0.4 * (ms_grid - ms_star))
    pdf = L ** (alpha_s + 1) * np.exp(-L)
    pdf /= np.trapz(pdf, ms_grid)
    return pdf


# -----------------------------------------------------------------------------
# Main 6D A(eta) computation (sigma_gamma free)
# -----------------------------------------------------------------------------


def build_eta_grid():
    """Return default 6D grids for ``mu_DM``, ``beta_DM``, ``sigma_DM``, ``mu_gamma``, ``sigma_gamma``, ``alpha``.

    This extends the previous 5D setup by promoting ``sigma_gamma`` to a free
    dimension participating in the A(eta) integral.
    """

    N = 50

    mu_DM_grid = np.linspace(12, 14, N)
    beta_DM_grid = np.linspace(1, 3, N)
    sigma_DM_grid = np.linspace(0.01, 0.6, N)
    mu_gamma_grid = np.linspace(0.5, 1.5, N)
    sigma_gamma_grid = np.linspace(0.01, 0.3, N)
    alpha_grid = np.linspace(-0.3, 0.3, N)

    return mu_DM_grid, beta_DM_grid, sigma_DM_grid, mu_gamma_grid, sigma_gamma_grid, alpha_grid

# 'mu_h0': 12.91,
# 'beta_h': 2.04,
# 'xi_h': 0.0,
# 'sigma_h': 0.37

def compute_A_eta(
    n_samples: int = 10000,
    ms_points: int = 100,
    m_lim: float = 26.5,
    n_jobs: int | None = None,
):
    """Monte Carlo estimate of the 6D normalisation grid ``A(eta)``.

    Parameters are chosen to mirror the pseudocode provided in the
    documentation.  The outer loop iterates over ``alpha`` and for each sample
    draws ``(Msps, Re, u, Mh)``.  For each ``muDM`` the halo-mass relation is
    evaluated and accumulated.
    """

    samples = sample_lens_population(n_samples)

    # Source magnitude prior grid (kept for metadata; K table already folds this in)
    # ms_grid = np.linspace(20.0, 30.0, ms_points)
    # p_ms = ms_distribution(ms_grid)

    # 6D grid over (mu_DM, beta_DM, sigma_DM, mu_gamma, sigma_gamma, alpha)
    (
        mu_DM_grid,
        beta_DM_grid,
        sigma_DM_grid,
        mu_gamma_grid,
        sigma_gamma_grid,
        alpha_grid,
    ) = build_eta_grid()

    # Use float32 to reduce memory footprint for the large accumulator
    A_accum = np.zeros(
        (
            mu_DM_grid.size,
            beta_DM_grid.size,
            sigma_DM_grid.size,
            mu_gamma_grid.size,
            sigma_gamma_grid.size,
            alpha_grid.size,
        ),
        dtype=np.float32,
    )

    # Precompute p_gamma over all (mu_gamma, sigma_gamma) for every sample
    # Shape: (N_mu_gamma, N_sigma_gamma, n_samples)
    gamma_in = samples["gamma_in"][None, None, :]
    MU_G, SG_G = np.meshgrid(mu_gamma_grid, sigma_gamma_grid, indexing="ij")
    MU_G = MU_G[:, :, None]
    SG_G = SG_G[:, :, None]
    p_gamma_table = np.exp(-0.5 * ((gamma_in - MU_G) / SG_G) ** 2) / (SG_G * np.sqrt(2 * np.pi))
    # Ensure contiguity and float32 to speed einsum and save memory
    p_gamma_table = np.ascontiguousarray(p_gamma_table.astype(np.float32, copy=False))

    for a_idx, alpha in enumerate(tqdm(alpha_grid, desc="alpha loop")):
        # Mstar used in lensing is Msps + alpha
        logM_star = samples["logM_star_sps"] + alpha

        mu1, mu2, betamax = compute_magnifications(
            logM_star,
            samples["logRe"],
            samples["logMh"],
            samples["gamma_in"],
            samples["beta"],
            samples["zl"],
            samples["zs"],
            n_jobs=n_jobs,
        )



        valid = (mu1 > 0) & (mu2 > 0) & (betamax > 0)

        # Photometric/detection term integrated over source magnitude (from prebuilt K table)
        T1 = K_interp(mu1[valid], mu2[valid])

        # ---- T2: source position weighting ----
        T2 = betamax[valid] ** 2

        # Combined static weight per Monte Carlo sample
        w_static = (T1 * T2).astype(np.float32, copy=False)

        # ---- T3: halo-mass + gamma relation for each grid point (vectorized, batched) ----
        valid_idx = np.where(valid)[0]

        if valid_idx.size == 0:
            continue

        # Prepare grids with explicit singleton axes for broadcasting
        mu_DM = mu_DM_grid[:, None, None, None].astype(np.float32, copy=False)
        beta_DM = beta_DM_grid[None, :, None, None].astype(np.float32, copy=False)
        sigma_DM = sigma_DM_grid[None, None, :, None].astype(np.float32, copy=False)

        batch_size = 256  # controls memory usage
        # HOTSPOT: vectorized accumulation
        for start in tqdm(range(0, valid_idx.size, batch_size),
                          desc="accumulate batches", leave=False):
            end = min(start + batch_size, valid_idx.size)
            idx_b = valid_idx[start:end]

            # Gather batch data and cast to float32
            logM_sps_b = samples["logM_star_sps"][idx_b].astype(np.float32, copy=False)
            logMh_b = samples["logMh"][idx_b].astype(np.float32, copy=False)
            w_b = w_static[start:end].astype(np.float32, copy=False)

            # Skip empty or non-positive weights quickly
            if not np.any(np.isfinite(w_b) & (w_b > 0)):
                continue

            # p_Mh over (mu, beta, sigma, batch)
            delta_b = (logM_sps_b - 11.4)[None, None, None, :]
            mean_Mh_b = mu_DM + beta_DM * delta_b  # (Nmu,Nbeta,1,B)
            logMh_b4 = logMh_b[None, None, None, :]  # (1,1,1,B)
            z = (logMh_b4 - mean_Mh_b) / sigma_DM  # broadcasts to (Nmu,Nbeta,Nsigma,B)
            p_Mh = np.exp(-0.5 * z.astype(np.float32) ** 2) / (
                sigma_DM * np.float32(np.sqrt(2.0 * np.pi))
            )  # (Nmu,Nbeta,Nsigma,B)

            # p_gamma for batch (mu_gamma, sigma_gamma, B)
            p_gamma_b = p_gamma_table[:, :, idx_b]  # (Nug,Nsg,B)

            # Batched einsum over sample axis 'b'
            # contrib shape: (Nmu, Nbeta, Nsigma, Nug, Nsg)
            contrib = np.einsum(
                'ijkb,uvb,b->ijkuv', p_Mh, p_gamma_b, w_b, optimize=True
            ).astype(np.float32, copy=False)

            A_accum[:, :, :, :, :, a_idx] += contrib



    Mh_range = samples["logMh_max"] - samples["logMh_min"]
    A = Mh_range * A_accum / n_samples

    # ---- Write to HDF5 (6D: mu, beta, sigma, mu_gamma, sigma_gamma, alpha) ----
    out_dir = os.path.join(os.path.dirname(__file__), "aeta_tables")
    os.makedirs(out_dir, exist_ok=True)
    fname = (
        f"Aeta6D_mu{mu_DM_grid.size}_beta{beta_DM_grid.size}"
        f"_sigma{sigma_DM_grid.size}_mugamma{mu_gamma_grid.size}_sigmagamma{sigma_gamma_grid.size}_alpha{alpha_grid.size}.h5"
    )
    out_path = os.path.join(out_dir, fname)

    with h5py.File(out_path, "w") as f:
        # Metadata group
        gmeta = f.create_group("metadata")
        gmeta.attrs["scatter_mag"] = float(SCATTER.mag)
        gmeta.attrs["scatter_star"] = float(SCATTER.star)
        gmeta.attrs["m_lim"] = float(m_lim)
        gmeta.attrs["alpha_s"] = float(-1.3)
        gmeta.attrs["m_s_star"] = float(24.5)
        gmeta.attrs["n_samples"] = int(n_samples)
        gmeta.attrs["K_table"] = os.path.basename(
            os.path.join(os.path.dirname(__file__), "K_mu0_10000_midRes_K_table_mu50000_ms10000.h5")
        )

        # Grids and A table (6D)
        g = f.create_group("grids")
        g.create_dataset("mu_DM_grid", data=mu_DM_grid)
        g.create_dataset("beta_DM_grid", data=beta_DM_grid)
        g.create_dataset("sigma_DM_grid", data=sigma_DM_grid)
        g.create_dataset("mu_gamma_grid", data=mu_gamma_grid)
        g.create_dataset("sigma_gamma_grid", data=sigma_gamma_grid)
        g.create_dataset("alpha_grid", data=alpha_grid)
        g.create_dataset("A_grid", data=A.astype(np.float32), compression="gzip")

        # Optional cache of Monte Carlo samples for reproducibility
        gcache = f.create_group("cache")
        gcache.create_dataset("logM_star_sps", data=samples["logM_star_sps"], compression="gzip")
        gcache.create_dataset("logRe", data=samples["logRe"], compression="gzip")
        gcache.create_dataset("logMh", data=samples["logMh"], compression="gzip")
        gcache.create_dataset("beta_unit", data=samples["beta"], compression="gzip")
        gcache.create_dataset("gamma_in", data=samples["gamma_in"], compression="gzip")
        gcache.attrs["zl"] = float(samples["zl"]) if np.isscalar(samples["zl"]) else float(np.asarray(samples["zl"]).ravel()[0])
        gcache.attrs["zs"] = float(samples["zs"]) if np.isscalar(samples["zs"]) else float(np.asarray(samples["zs"]).ravel()[0])

    return out_path


# Provide a convenience alias reflecting the terminology in the documentation
# normfactor = compute_A_eta


def load_A_eta_interpolator(path: str):
    """Load an interpolator for A(η) from HDF5.

    Preferred: 6D grid (mu_DM, beta_DM, sigma_DM, mu_gamma, sigma_gamma, alpha).
    Backward compatible: 5D grid without sigma_gamma; returns a wrapper that
    accepts 6D inputs and ignores the sigma_gamma column.
    """

    import h5py
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    with h5py.File(path, "r") as f:
        mu_DM_grid = np.array(f["grids/mu_DM_grid"])
        beta_DM_grid = np.array(f["grids/beta_DM_grid"])
        sigma_DM_grid = np.array(f["grids/sigma_DM_grid"])
        A_grid = np.array(f["grids/A_grid"])  # may be 5D or 6D

        # Try to read gamma grids; sigma_gamma may not exist for legacy 5D
        mu_gamma_grid = np.array(f["grids/mu_gamma_grid"]) if "grids/mu_gamma_grid" in f else None
        sigma_gamma_grid = (
            np.array(f["grids/sigma_gamma_grid"]) if "grids/sigma_gamma_grid" in f else None
        )
        alpha_grid = np.array(f["grids/alpha_grid"]) if "grids/alpha_grid" in f else None

    if A_grid.ndim == 6 and (sigma_gamma_grid is not None):
        return RegularGridInterpolator(
            (mu_DM_grid, beta_DM_grid, sigma_DM_grid, mu_gamma_grid, sigma_gamma_grid, alpha_grid),
            A_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    if A_grid.ndim == 5 and (mu_gamma_grid is not None) and (alpha_grid is not None):
        # Legacy 5D table: (mu, beta, sigma, mu_gamma, alpha)
        rgi5 = RegularGridInterpolator(
            (mu_DM_grid, beta_DM_grid, sigma_DM_grid, mu_gamma_grid, alpha_grid),
            A_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        class _Adapter6Dto5D:
            def __call__(self, pts):
                arr = np.asarray(pts)
                if arr.ndim == 1 and arr.size == 6:
                    mu, beta, sigma, mu_g, sig_g, alpha = map(float, arr)
                    return rgi5((mu, beta, sigma, mu_g, alpha))
                elif arr.ndim == 2 and arr.shape[-1] >= 6:
                    p5 = np.column_stack([arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 5]])
                    return rgi5(p5)
                else:
                    raise ValueError("A(eta) adapter expects points of shape (6,) or (N,6)")

        return _Adapter6Dto5D()

    raise ValueError("Unsupported A_grid dimensionality or missing grids in HDF5 file")


if __name__ == "__main__":
    compute_A_eta()
