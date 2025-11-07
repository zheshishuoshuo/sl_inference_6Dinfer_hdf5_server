"""Build K(mu1, mu2) and optional interpolator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, Tuple, Callable, Iterator
import numpy as np
import h5py
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from .utils import selection_function


@dataclass
class KTableConfig:
    """Configuration for mu and ms grids."""
    mu_min: float
    mu_max: float
    n_mu: int
    grid_type: Literal["logspace", "linspace"] = "logspace"
    ms_min: float = 20.0
    ms_max: float = 30.0
    n_ms: int = 2000
    m_lim: float = 26.5
    scatter_mag: float = 0.1


def build_mu_grid(cfg: KTableConfig) -> np.ndarray:
    """Build mu grid using logspace or linspace."""
    if cfg.grid_type == "logspace":
        if cfg.mu_min <= 0:
            raise ValueError("mu_min must be > 0 for logspace")
        return np.logspace(np.log10(cfg.mu_min), np.log10(cfg.mu_max), cfg.n_mu)
    if cfg.grid_type == "linspace":
        return np.linspace(cfg.mu_min, cfg.mu_max, cfg.n_mu)
    raise ValueError("grid_type must be 'logspace' or 'linspace'")


def source_mag_prior(ms_grid: np.ndarray, alpha_s: float = -1.3, ms_star: float = 24.5) -> np.ndarray:
    """Normalized prior p(ms) over source magnitude."""
    ms = np.asarray(ms_grid, dtype=float)
    L = 10.0 ** (-0.4 * (ms - ms_star))
    pdf = L ** (alpha_s + 1.0) * np.exp(-L)
    norm = np.trapz(pdf, ms)
    return pdf / norm


def compute_p_det(
    mu_grid: np.ndarray,
    ms_grid: np.ndarray,
    m_lim: float,
    scatter_mag: float,
    batch_size: int,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute p_det(mu, ms) in batches, optionally threaded."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_mu = len(mu_grid)
    P = np.empty((n_mu, len(ms_grid)), dtype=np.float32)
    tasks = [(s, min(s + batch_size, n_mu)) for s in range(0, n_mu, batch_size)]

    def _compute_slice(start: int, stop: int) -> tuple[int, int, np.ndarray]:
        mu_b = mu_grid[start:stop]
        p_det = selection_function(mu_b[:, None], m_lim, ms_grid[None, :], scatter_mag)
        p_det = np.nan_to_num(p_det, nan=0.0, posinf=0.0, neginf=0.0)
        return start, stop, p_det.astype(np.float32, copy=False)

    if n_jobs and n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_compute_slice, s, e) for s, e in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="computing p_det"):
                s, e, arr = fut.result()
                P[s:e, :] = arr
    else:
        for s, e in tqdm(tasks, desc="computing p_det"):
            _, _, arr = _compute_slice(s, e)
            P[s:e, :] = arr

    return P


def generate_K_blocks(
    P: np.ndarray,
    p_ms: np.ndarray,
    ms_grid: np.ndarray,
    batch_size: int,
) -> Iterator[tuple[int, int, int, int, np.ndarray]]:
    """Yield symmetric K blocks (i, j, k, l, block)."""
    n_mu = P.shape[0]
    P64 = P.astype(np.float64, copy=False)
    p_ms64 = p_ms.astype(np.float64, copy=False)

    for i in range(0, n_mu, batch_size):
        j = min(i + batch_size, n_mu)
        Ai = P64[i:j, :] * p_ms64[None, :]
        for k in range(0, i + 1, batch_size):
            l = min(k + batch_size, n_mu)
            Bj = P64[k:l, :]
            bi = j - i
            bj = l - k
            block = np.empty((bi, bj), dtype=np.float64)
            for ii in range(bi):
                block[ii, :] = np.trapz(Bj * Ai[ii, None, :], x=ms_grid, axis=1)
            yield i, j, k, l, block


def write_K_to_hdf5(
    block_iter: Iterator[tuple[int, int, int, int, np.ndarray]],
    n_mu: int,
    batch_size: int,
    save_path: str,
) -> None:
    """Write K blocks to HDF5 with symmetry."""
    with h5py.File(save_path, "w") as f:
        dset = f.create_dataset(
            "/K_table",
            shape=(n_mu, n_mu),
            dtype=np.float32,
            chunks=(min(batch_size, n_mu), n_mu),
            compression="gzip",
            compression_opts=4,
        )
        for i, j, k, l, block in tqdm(block_iter, desc="writing K [HDF5]"):
            block32 = block.astype(np.float32, copy=False)
            dset[i:j, k:l] = block32
            dset[k:l, i:j] = block32.T
        f.flush()


def build_K_table(
    mu_min: float,
    mu_max: float,
    n_mu: int,
    *,
    grid_type: Literal["logspace", "linspace"] = "logspace",
    ms_min: float = 20.0,
    ms_max: float = 30.0,
    n_ms: int = 400,
    m_lim: float = 26.5,
    scatter_mag: float = 0.1,
    batch: int | None = None,
    n_jobs: int | None = 1,
    save_path: str | None = None,
    return_arrays: bool = True,
) -> Tuple[np.ndarray, np.ndarray | str]:
    """Build K table; return arrays or write HDF5."""
    cfg = KTableConfig(
        mu_min=mu_min,
        mu_max=mu_max,
        n_mu=n_mu,
        grid_type=grid_type,
        ms_min=ms_min,
        ms_max=ms_max,
        n_ms=n_ms,
        m_lim=m_lim,
        scatter_mag=scatter_mag,
    )

    mu_grid = build_mu_grid(cfg).astype(np.float64)
    ms_grid = np.linspace(ms_min, ms_max, n_ms, dtype=np.float64)
    p_ms = source_mag_prior(ms_grid).astype(np.float64)

    batch_size = int(batch) if batch and batch > 0 else max(1, min(1024, n_mu))

    P = compute_p_det(mu_grid, ms_grid, m_lim, scatter_mag, batch_size, n_jobs or 1)

    if save_path and not return_arrays:
        block_iter = generate_K_blocks(P, p_ms, ms_grid, batch_size)
        write_K_to_hdf5(block_iter, n_mu, batch_size, save_path)
        return mu_grid.astype(np.float32, copy=False), save_path

    K = np.zeros((n_mu, n_mu), dtype=np.float64)
    for i, j, k, l, block in tqdm(
        generate_K_blocks(P, p_ms, ms_grid, batch_size),
        total=(n_mu // batch_size) * (n_mu // batch_size + 1) // 2,
        desc="accumulating K matrix",
    ):
        K[i:j, k:l] = block
        K[k:l, i:j] = block.T

    return mu_grid.astype(np.float32, copy=False), K.astype(np.float32, copy=False)




def save_K_table_hdf5(
    path: str,
    mu_grid: np.ndarray,
    K: np.ndarray,
    *,
    attrs: dict | None = None,
) -> None:
    """Save mu_grid and K to HDF5."""
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "/mu_grid",
            data=mu_grid.astype(np.float32, copy=False),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "/K_table",
            data=K.astype(np.float32, copy=False),
            compression="gzip",
            compression_opts=4,
        )
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v




# --------------------------# K(mu1, mu2) interpolator --------------------------#


def build_K_interpolator_from_arrays(
    mu_grid: np.ndarray,
    K: np.ndarray,
    *,
    method: Literal["linear", "nearest"] = "linear",
    bounds_error: bool = False,
    fill_value: float | None = 0.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return K(mu1, mu2) interpolator over mu_grid."""
    rgi = RegularGridInterpolator(
        (np.asarray(mu_grid), np.asarray(mu_grid)),
        np.asarray(K, dtype=np.float64),
        method=method,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )

    def interp(mu1: np.ndarray | float, mu2: np.ndarray | float) -> np.ndarray:
        a = np.asarray(mu1, dtype=float)
        b = np.asarray(mu2, dtype=float)
        A, B = np.broadcast_arrays(a, b)
        pts = np.column_stack([A.ravel(), B.ravel()])
        out = rgi(pts).reshape(A.shape)
        return out.astype(np.float32, copy=False)

    return interp


def load_K_interpolator(
    path: str,
    *,
    method: Literal["linear", "nearest"] = "linear",
    bounds_error: bool = False,
    fill_value: float | None = 0.0,
    return_arrays: bool = False,
):
    """Load K table from HDF5 and build interpolator."""
    with h5py.File(path, "r") as f:
        mu = np.array(f["/mu_grid"], dtype=np.float32)
        K = np.array(f["/K_table"], dtype=np.float32)
    interp = build_K_interpolator_from_arrays(
        mu, K, method=method, bounds_error=bounds_error, fill_value=fill_value
    )
    if return_arrays:
        return interp, mu, K
    return interp


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Build K(mu1, mu2) and save to HDF5")
    p.add_argument("--mu-min", type=float, default=1e-4, help="mu grid minimum")
    p.add_argument("--mu-max", type=float, default=500.0, help="mu grid maximum")
    p.add_argument("--n-mu", type=int, default=1000, help="number of mu grid points")
    p.add_argument("--grid-type", choices=["logspace", "linspace"], default="logspace", help="mu grid spacing")
    p.add_argument("--ms-min", type=float, default=20.0, help="source magnitude min")
    p.add_argument("--ms-max", type=float, default=30.0, help="source magnitude max")
    p.add_argument("--n-ms", type=int, default=2000, help="number of ms grid points")
    p.add_argument("--m-lim", type=float, default=26.5, help="survey limiting magnitude")
    p.add_argument("--scatter-mag", type=float, default=0.1, help="photometric scatter (mag)")
    p.add_argument("--prefix", type=str, default="K", help="filename prefix")
    p.add_argument("--batch", type=int, default=0, help="batch size for p_det; 0=auto")
    p.add_argument("--n-jobs", type=int, default=1, help="number of threads for p_det computation")
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    mu_grid, K = build_K_table(
        args.mu_min,
        args.mu_max,
        args.n_mu,
        grid_type=args.grid_type,
        ms_min=args.ms_min,
        ms_max=args.ms_max,
        n_ms=args.n_ms,
        m_lim=args.m_lim,
        scatter_mag=args.scatter_mag,
        batch=args.batch,
        n_jobs=args.n_jobs,
    )

    out_name = f"{args.prefix}_K_table_mu{mu_grid.size}_ms{args.n_ms}.h5"

    attrs = {
        "mu_min": float(args.mu_min),
        "mu_max": float(args.mu_max),
        "n_mu": int(mu_grid.size),
        "ms_min": float(args.ms_min),
        "ms_max": float(args.ms_max),
        "n_ms": int(args.n_ms),
        "m_lim": float(args.m_lim),
        "scatter_mag": float(args.scatter_mag),
        "grid_type": str(args.grid_type),
        "alpha_s": float(-1.3),
        "m_s_star": float(24.5),
    }

    save_K_table_hdf5(out_name, mu_grid, K, attrs=attrs)
    print(f"Saved to {out_name}")
    print(f"K shape: {K.shape}")


if __name__ == "__main__":
    main()
