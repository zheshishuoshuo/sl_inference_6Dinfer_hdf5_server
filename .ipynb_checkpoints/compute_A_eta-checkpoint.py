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

    N = 30

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
    # Truncated-normal on [0, 2]: pdf / Z where Z = CDF(2) - CDF(0)
    pdf = np.exp(-0.5 * ((gamma_in - MU_G) / SG_G) ** 2) / (SG_G * np.sqrt(2 * np.pi))
    Z = norm.cdf(2.0, loc=MU_G, scale=SG_G) - norm.cdf(0.0, loc=MU_G, scale=SG_G)
    Z = np.clip(Z, 1e-12, None)
    p_gamma_table = pdf / Z
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

    # 所有写入结束后，添加以下内容：
    import gc, sys
    gc.collect()
    print("Computation done. Exiting.", flush=True)
    sys.exit(0)

    return out_path  # ← 这行可以删掉或保留，但不会被执行


# Provide a convenience alias reflecting the terminology in the documentation
# normfactor = compute_A_eta


def load_A_eta_interpolator(path: str):
    """Load a local, on-demand interpolator for A(η) from HDF5.

    - Supports 6D tables: (mu_DM, beta_DM, sigma_DM, mu_gamma, sigma_gamma, alpha).
    - Backwards compatible with 5D tables that omit sigma_gamma. In that case,
      inputs are still 6D but the sigma_gamma coordinate is ignored.

    This implementation avoids loading the full A_grid into memory. Instead, it
    loads a small local block (up to 5 points per dimension) around each query
    using h5py slicing and performs linear interpolation via
    scipy.interpolate.RegularGridInterpolator on that block. A simple last-block
    cache accelerates repeated queries in the same neighborhood.
    """

    import numpy as np
    import h5py
    from scipy.interpolate import RegularGridInterpolator

    class LocalAInterpolator:
        def __init__(self, h5path: str, window: int = 2):
            self._h5path = h5path
            self._file = h5py.File(h5path, "r")
            self._dset = self._file["grids/A_grid"]
            self._pid = os.getpid()

            # Grids (always load into memory; small compared to A_grid)
            self.mu_DM_grid = np.array(self._file["grids/mu_DM_grid"])  # (Nmu,)
            self.beta_DM_grid = np.array(self._file["grids/beta_DM_grid"])  # (Nbeta,)
            self.sigma_DM_grid = np.array(self._file["grids/sigma_DM_grid"])  # (Nsigma,)
            self.mu_gamma_grid = (
                np.array(self._file["grids/mu_gamma_grid"]) if "grids/mu_gamma_grid" in self._file else None
            )
            self.sigma_gamma_grid = (
                np.array(self._file["grids/sigma_gamma_grid"]) if "grids/sigma_gamma_grid" in self._file else None
            )
            self.alpha_grid = np.array(self._file["grids/alpha_grid"]) if "grids/alpha_grid" in self._file else None

            # Determine dimensionality and mapping from 6D input -> table dims
            if self._dset.ndim == 6 and self.sigma_gamma_grid is not None:
                self._dims = 6
                # Order in table
                self._grid_list = [
                    self.mu_DM_grid,
                    self.beta_DM_grid,
                    self.sigma_DM_grid,
                    self.mu_gamma_grid,
                    self.sigma_gamma_grid,
                    self.alpha_grid,
                ]
                # Indices to take from a 6D input point
                self._input_to_table_idx = (0, 1, 2, 3, 4, 5)
            elif self._dset.ndim == 5 and (self.mu_gamma_grid is not None) and (self.alpha_grid is not None):
                # Legacy 5D: (mu, beta, sigma, mu_gamma, alpha)
                self._dims = 5
                self._grid_list = [
                    self.mu_DM_grid,
                    self.beta_DM_grid,
                    self.sigma_DM_grid,
                    self.mu_gamma_grid,
                    self.alpha_grid,
                ]
                # Map from 6D input -> 5D table (skip sigma_gamma)
                self._input_to_table_idx = (0, 1, 2, 3, 5)
            else:
                raise ValueError("Unsupported A_grid dimensionality or missing grids in HDF5 file")

            # Local window size on each side; target block length up to 2*window+1 (i.e. 5)
            self.window = int(window)

            # Simple last-block cache
            self._cache_key = None  # tuple of ( (start, stop), ... )
            self._cache_interpolator = None  # RegularGridInterpolator instance
            self._cache_grids = None  # list of 1D arrays for each dim in block

        def close(self):
            try:
                if self._file:
                    self._file.close()
            except Exception:
                pass

        def __del__(self):
            self.close()

        def _nearest_center_index(self, grid: np.ndarray, v: float) -> int:
            # Find nearest grid index to v
            i = int(np.searchsorted(grid, v, side="left"))
            if i <= 0:
                return 0
            if i >= grid.size:
                return grid.size - 1
            # Compare neighbors
            left = i - 1
            if abs(v - grid[left]) <= abs(grid[i] - v):
                return left
            return i

        def _block_bounds(self, grid: np.ndarray, center: int) -> tuple[int, int]:
            # Compute [start, stop) for a block centered at index 'center'
            n = grid.size
            start = max(center - self.window, 0)
            stop = min(center + self.window + 1, n)
            # Ensure at least 2 points for linear interpolation
            if stop - start < 2:
                if stop < n:
                    stop = min(start + 2, n)
                else:
                    start = max(n - 2, 0)
            return start, stop

        def _slices_for_point(self, pt6: np.ndarray) -> tuple:
            # Map 6D input to table's coordinate order and compute per-dim slices
            coords = [float(pt6[i]) for i in self._input_to_table_idx]
            starts_stops = []
            for g, v in zip(self._grid_list, coords):
                c = self._nearest_center_index(g, v)
                s, e = self._block_bounds(g, c)
                starts_stops.append((s, e))
            return tuple(slice(s, e) for (s, e) in starts_stops), tuple(starts_stops)

        def _get_interpolator_for_slices(self, slices: tuple, key: tuple):
            # Check cache
            if self._cache_key == key and self._cache_interpolator is not None:
                return self._cache_interpolator, self._cache_grids

            # Load local block from HDF5
            sub_grids = [g[s] for g, s in zip(self._grid_list, slices)]
            sub_vals = self._dset[slices]

            rgi = RegularGridInterpolator(
                tuple(sub_grids),
                sub_vals,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )

            # Update cache
            self._cache_key = key
            self._cache_interpolator = rgi
            self._cache_grids = sub_grids
            return rgi, sub_grids

        def __call__(self, pts):
            # Reopen HDF5 file in child processes if needed (safe for multiprocessing)
            cur_pid = os.getpid()
            if cur_pid != self._pid:
                try:
                    # Drop cached state relying on the previous file handle
                    if self._file:
                        try:
                            self._file.close()
                        except Exception:
                            pass
                    self._file = h5py.File(self._h5path, "r")
                    self._dset = self._file["grids/A_grid"]
                    self._pid = cur_pid
                    # Invalidate cache as underlying objects changed
                    self._cache_key = None
                    self._cache_interpolator = None
                    self._cache_grids = None
                except Exception:
                    # If reopen fails, proceed and let h5py raise later on first access
                    self._pid = cur_pid

            arr = np.asarray(pts, dtype=float)
            if arr.ndim == 1:
                if arr.size < 6:
                    raise ValueError("A(eta) expects points of shape (6,) or (N,6)")
                arr2 = arr.reshape(1, -1)
                single = True
            elif arr.ndim == 2 and arr.shape[1] >= 6:
                arr2 = arr
                single = False
            else:
                raise ValueError("A(eta) expects points of shape (6,) or (N,6)")

            # Group points by their local block (slices)
            keys = []
            slices_list = []
            for p in arr2:
                slc, key = self._slices_for_point(p)
                slices_list.append(slc)
                keys.append(key)

            results = np.empty(arr2.shape[0], dtype=float)

            # Process groups sharing the same block
            # Build mapping: key -> indices
            from collections import defaultdict

            groups = defaultdict(list)
            for i, k in enumerate(keys):
                groups[k].append(i)

            for k, idxs in groups.items():
                slc = slices_list[idxs[0]]
                rgi, _ = self._get_interpolator_for_slices(slc, k)

                # Prepare local coordinates for this block
                pts_local = []
                for i in idxs:
                    # select the relevant dims from the 6D input according to table mapping
                    if self._dims == 6:
                        coord = arr2[i, [0, 1, 2, 3, 4, 5]]
                    else:  # 5D legacy, drop sigma_gamma (index 4)
                        coord = arr2[i, [0, 1, 2, 3, 5]]
                    pts_local.append(coord)
                pts_local = np.asarray(pts_local, dtype=float)

                vals = rgi(pts_local)
                results[idxs] = vals

            return results[0] if single else results

    return LocalAInterpolator(path)


if __name__ == "__main__":
    compute_A_eta()
