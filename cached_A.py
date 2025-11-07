import os
import glob
import numpy as np
import h5py
from typing import Any
from scipy.interpolate import RegularGridInterpolator

# Default betaDM value retained for legacy collapse helper; not used by default
_DEFAULT_BETA_DM = 2.04


def _build_interpolator_from_arrays(
    mu_grid: np.ndarray,
    sigma_grid: np.ndarray,
    alpha_grid: np.ndarray,
    A_grid: np.ndarray,
) -> RegularGridInterpolator:
    # Unified 3D RGI with NaN fill on out-of-bounds
    return RegularGridInterpolator(
        (np.asarray(mu_grid), np.asarray(sigma_grid), np.asarray(alpha_grid)),
        np.asarray(A_grid),
        bounds_error=False,
        fill_value=np.nan,
    )


def _read_first_existing(f: h5py.File, candidates: list[str]) -> np.ndarray | h5py.Dataset:
    """Return the first existing dataset (as h5py.Dataset) or raise KeyError.

    Accepts a list of candidate absolute paths within the HDF5 file. If a
    candidate exists, returns the dataset object; otherwise raises KeyError.
    """
    for key in candidates:
        try:
            if key in f:
                return f[key]
        except Exception:
            continue
    raise KeyError(f"None of the datasets exist: {candidates}")


def _collapse_4d_to_3d(
    mu: np.ndarray,
    beta: np.ndarray,
    sigma: np.ndarray,
    alpha: np.ndarray,
    A4: np.ndarray,
    *,
    beta_fixed: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse a 4D (mu, beta, sigma, alpha) grid to 3D (mu, sigma, alpha).

    Rules:
    - If beta has length 1: drop beta axis directly.
    - Else if sigma has length 1: evaluate along beta at ``beta_fixed`` and
      keep a singleton sigma axis to remain compatible with callers.
    - Else: evaluate along beta at ``beta_fixed`` for all (mu, sigma, alpha).

    Returns (mu_grid, sigma_grid, alpha_grid, A3) suitable for 3D RGI.
    """
    mu = np.asarray(mu)
    beta = np.asarray(beta)
    sigma = np.asarray(sigma)
    alpha = np.asarray(alpha)
    A4 = np.asarray(A4)

    # Special case: both beta and sigma are singleton -> squeeze both
    if beta.size == 1 and sigma.size == 1:
        # A2: (n_mu, n_alpha)
        A2 = A4[:, 0, 0, :]
        # Return 3D with a placeholder singleton sigma grid [0.0]
        return mu, np.array([0.0], dtype=float), alpha, A2[:, None, :]

    if beta.size == 1:
        A3 = A4[:, 0, :, :]  # (n_mu, n_sigma, n_alpha)
        return mu, sigma, alpha, A3

    # Determine which beta to use when collapsing
    b0 = _DEFAULT_BETA_DM if beta_fixed is None else float(beta_fixed)

    if sigma.size == 1:
        # Slice sigma, interpolate over (mu, beta, alpha) at beta=b0
        A_mba = A4[:, :, 0, :]  # (n_mu, n_beta, n_alpha)
        # Build RGI over (mu, beta, alpha) to evaluate at (mu, b0, alpha)
        rgi = RegularGridInterpolator(
            (mu, beta, alpha), A_mba, bounds_error=False, fill_value=np.nan
        )
        MU, AL = np.meshgrid(mu, alpha, indexing="ij")
        pts = np.column_stack([MU.reshape(-1), np.full(MU.size, b0), AL.reshape(-1)])
        vals = rgi(pts).reshape(mu.size, alpha.size)
        # Reinsert singleton sigma axis
        A3 = vals[:, None, :]
        return mu, sigma, alpha, A3

    # General case: collapse beta dimension at b0, keep full sigma
    # Build RGI over (mu, beta, sigma, alpha) and evaluate at beta=b0
    rgi4 = RegularGridInterpolator(
        (mu, beta, sigma, alpha), A4, bounds_error=False, fill_value=np.nan
    )
    MU, SG, AL = np.meshgrid(mu, sigma, alpha, indexing="ij")
    pts = np.column_stack([
        MU.reshape(-1),
        np.full(MU.size, b0),
        SG.reshape(-1),
        AL.reshape(-1),
    ])
    vals = rgi4(pts).reshape(mu.size, sigma.size, alpha.size)
    return mu, sigma, alpha, vals


class _OneD_AetaInterpolator:
    """Callable wrapper to mimic RegularGridInterpolator for 1D A(mu_DM).

    Accepts inputs shaped like ``(N, 3)`` where only the first column (mu_DM)
    is used; the remaining columns (sigma_DM, alpha) are ignored.
    Returns an array of shape ``(N,)`` with ``NaN`` outside bounds.
    """

    def __init__(self, mu_grid: np.ndarray, A_grid: np.ndarray) -> None:
        self.mu_grid = np.asarray(mu_grid, dtype=float)
        self.A_grid = np.asarray(A_grid, dtype=float)
        # Expose a grid tuple for compatibility with callers that inspect ndim
        self.grid = (self.mu_grid,)

    def __call__(self, points: Any) -> np.ndarray:
        arr = np.asarray(points)
        # Handle scalar mu
        if arr.ndim == 0:
            mu = np.array([float(arr)])
        elif arr.ndim == 1:
            # Could be (3,) -> take first; or (N,) -> treat as mu vector
            mu = np.array([float(arr[0])]) if arr.size in (3,) else arr.astype(float)
        else:
            # Expect shape (N, 3); take first column as mu
            if arr.shape[-1] >= 1:
                mu = arr[..., 0].astype(float).reshape(-1)
            else:
                raise ValueError("Input points must have at least one column for mu_DM")

        # Linear 1D interpolation with NaN outside bounds
        left = np.nan
        right = np.nan
        vals = np.interp(mu, self.mu_grid, self.A_grid, left=left, right=right)
        return vals


def cached_A_interp(
    scatter_mag: float,
    scatter_star: float,
    directory: str = "aeta_tables",
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    interactive: bool = False,
) -> RegularGridInterpolator:
    """Return an A(eta) interpolator from the local aeta_tables directory.

    Notes
    -----
    - Ignores ``scatter_mag`` and ``scatter_star`` when choosing files.
    - Always searches the subdirectory ``aeta_tables/`` next to this module.
    - If multiple ``*.h5`` files exist there, sorts by filename and selects
      the last one (lexicographically latest).
    - Supports 1D (mu), 3D (mu, sigma, alpha), 4D (mu, beta, sigma, alpha),
      and 6D (mu, beta, sigma, alpha, mu_gamma, sigma_gamma) tables. If a 6D
      table is found, return a 6D interpolator directly with matching axis order.
      Legacy 1D/3D/4D are preserved.
    """

    base_dir = os.path.dirname(__file__)
    # Per request, always search local aeta_tables under this module
    search_dir = os.path.join(base_dir, "aeta_tables")
    # Pick up any Aeta tables in the subdir
    files = sorted(glob.glob(os.path.join(search_dir, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No A(eta) HDF5 files found under: {search_dir}")

    # Pick the lexicographically latest file by name
    chosen_path = files[-1]

    # Interactive selection if requested and running on TTY
    if interactive:
        import sys
        if sys.stdin.isatty():
            rows = []
            for p in files:
                try:
                    with h5py.File(p, "r") as f:
                        # Metadata (best-effort)
                        try:
                            md = f["metadata"].attrs
                        except Exception:
                            md = f.attrs
                        # Grids (try multiple locations)
                        mu_d = _read_first_existing(f, [
                            "grids/mu_DM_grid",
                            "/mu_dm_grid",
                        ])
                        mu_len = int(getattr(mu_d, "shape", [0])[0])
                        # Detect dimensionality and grid lengths
                        try:
                            A_d = _read_first_existing(f, [
                                "grids/A_grid",
                                "/A_grid",
                                "/A_eta",
                            ])
                            a_nd = int(getattr(A_d, "ndim", 0))
                        except Exception:
                            a_nd = 0
                        sigma_len = 1
                        alpha_len = 1
                        beta_len = 0
                        mu_gamma_len = 0
                        sigma_gamma_len = 0
                        if a_nd == 6:
                            try:
                                beta_len = int(_read_first_existing(f, [
                                    "grids/beta_DM_grid",
                                    "/beta_dm_grid",
                                ]).shape[0])
                            except Exception:
                                beta_len = 0
                            try:
                                sigma_len = int(_read_first_existing(f, [
                                    "grids/sigma_DM_grid",
                                    "/sigma_dm_grid",
                                ]).shape[0])
                            except Exception:
                                sigma_len = 1
                            try:
                                alpha_len = int(_read_first_existing(f, [
                                    "grids/alpha_grid",
                                    "/alpha_grid",
                                ]).shape[0])
                            except Exception:
                                alpha_len = 1
                            try:
                                mu_gamma_len = int(_read_first_existing(f, [
                                    "/mu_gamma_grid",
                                ]).shape[0])
                            except Exception:
                                mu_gamma_len = 0
                            try:
                                sigma_gamma_len = int(_read_first_existing(f, [
                                    "/sigma_gamma_grid",
                                ]).shape[0])
                            except Exception:
                                sigma_gamma_len = 0
                        elif a_nd == 4:
                            try:
                                beta_len = int(_read_first_existing(f, [
                                    "grids/beta_DM_grid",
                                    "/beta_dm_grid",
                                ]).shape[0])
                            except Exception:
                                beta_len = 0
                            try:
                                sigma_len = int(_read_first_existing(f, [
                                    "grids/sigma_DM_grid",
                                    "/sigma_dm_grid",
                                ]).shape[0])
                            except Exception:
                                sigma_len = 1
                            try:
                                alpha_len = int(_read_first_existing(f, [
                                    "grids/alpha_grid",
                                    "/alpha_grid",
                                ]).shape[0])
                            except Exception:
                                alpha_len = 1
                        elif a_nd == 3:
                            try:
                                sigma_len = int(_read_first_existing(f, [
                                    "grids/sigma_DM_grid",
                                    "/sigma_dm_grid",
                                ]).shape[0])
                            except Exception:
                                sigma_len = 1
                            try:
                                alpha_len = int(_read_first_existing(f, [
                                    "grids/alpha_grid",
                                    "/alpha_grid",
                                ]).shape[0])
                            except Exception:
                                alpha_len = 1
                        rows.append({
                            "path": p,
                            "mtime": os.path.getmtime(p),
                            "scatter_mag": float(md.get("scatter_mag", np.nan)),  # optional
                            "scatter_star": float(md.get("scatter_star", np.nan)),  # optional
                            "n_samples": int(md.get("n_samples", -1)),  # optional
                            "res": (mu_len, sigma_len, alpha_len) if (beta_len == 0 and mu_gamma_len == 0) else (
                                (mu_len, beta_len, sigma_len, alpha_len, mu_gamma_len, sigma_gamma_len) if mu_gamma_len > 0 else (mu_len, beta_len, sigma_len, alpha_len)
                            ),
                        })
                except Exception:
                    continue
            print("Available A(eta) tables:")
            for i, r in enumerate(rows):
                from datetime import datetime
                dt = datetime.utcfromtimestamp(r["mtime"]).strftime("%Y-%m-%dT%H:%M:%SZ")
                print(f"[{i}] {os.path.basename(r['path'])}  mtime={dt}  scatter_mag={r['scatter_mag']:.3g}  scatter_star={r['scatter_star']:.3g}  nsample={r['n_samples']}  res={r['res']}")
            try:
                idx = int(input("Select idx: ").strip())
                if 0 <= idx < len(rows):
                    chosen_path = rows[idx]["path"]
                    with h5py.File(chosen_path, "r") as f:
                        mu = np.asarray(_read_first_existing(f, [
                            "grids/mu_DM_grid",
                            "/mu_dm_grid",
                        ]))
                        Agr = _read_first_existing(f, [
                            "grids/A_grid",
                            "/A_grid",
                            "/A_eta",
                        ])
                        if Agr.ndim == 6:
                            beta = np.asarray(_read_first_existing(f, [
                                "grids/beta_DM_grid",
                                "/beta_dm_grid",
                            ]))
                            sigma = np.asarray(_read_first_existing(f, [
                                "grids/sigma_DM_grid",
                                "/sigma_dm_grid",
                            ]))
                            alpha = np.asarray(_read_first_existing(f, [
                                "grids/alpha_grid",
                                "/alpha_grid",
                            ]))
                            mu_g = np.asarray(_read_first_existing(f, [
                                "/mu_gamma_grid",
                            ]))
                            sg_g = np.asarray(_read_first_existing(f, [
                                "/sigma_gamma_grid",
                            ]))
                            A6 = np.asarray(Agr)
                            return RegularGridInterpolator(
                                (mu, beta, sigma, alpha, mu_g, sg_g), A6,
                                bounds_error=False,
                                fill_value=np.nan,
                            )
                        elif Agr.ndim == 4:
                            beta = np.asarray(_read_first_existing(f, [
                                "grids/beta_DM_grid",
                                "/beta_dm_grid",
                            ]))
                            sigma = np.asarray(_read_first_existing(f, [
                                "grids/sigma_DM_grid",
                                "/sigma_dm_grid",
                            ]))
                            alpha = np.asarray(_read_first_existing(f, [
                                "grids/alpha_grid",
                                "/alpha_grid",
                            ]))
                            A4 = np.asarray(Agr)
                            return RegularGridInterpolator(
                                (mu, beta, sigma, alpha), A4,
                                bounds_error=False,
                                fill_value=np.nan,
                            )
                        elif Agr.ndim == 3:
                            sigma = np.asarray(_read_first_existing(f, [
                                "grids/sigma_DM_grid",
                                "/sigma_dm_grid",
                            ]))
                            alpha = np.asarray(_read_first_existing(f, [
                                "grids/alpha_grid",
                                "/alpha_grid",
                            ]))
                            A = np.asarray(Agr)
                            return _build_interpolator_from_arrays(mu, sigma, alpha, A)
                        else:
                            # 1D legacy table
                            A1d = np.asarray(Agr)
                            return _OneD_AetaInterpolator(mu, A1d)
            except Exception:
                pass  # fall back to default matching below

    # Default: load the chosen file by name
    with h5py.File(chosen_path, "r") as f:
        # Prefer group attrs but fall back to file attrs (used only to read betaDM if present)
        try:
            md = f["metadata"].attrs
        except Exception:
            md = f.attrs

        mu = np.asarray(_read_first_existing(f, [
            "grids/mu_DM_grid",
            "/mu_dm_grid",
        ]))
        Agr = _read_first_existing(f, [
            "grids/A_grid",
            "/A_grid",
            "/A_eta",
        ])
        if Agr.ndim == 6:
            beta = np.asarray(_read_first_existing(f, [
                "grids/beta_DM_grid",
                "/beta_dm_grid",
            ]))
            sigma = np.asarray(_read_first_existing(f, [
                "grids/sigma_DM_grid",
                "/sigma_dm_grid",
            ]))
            alpha = np.asarray(_read_first_existing(f, [
                "grids/alpha_grid",
                "/alpha_grid",
            ]))
            mu_g = np.asarray(_read_first_existing(f, [
                "/mu_gamma_grid",
            ]))
            sg_g = np.asarray(_read_first_existing(f, [
                "/sigma_gamma_grid",
            ]))
            A6 = np.asarray(Agr)
            return RegularGridInterpolator(
                (mu, beta, sigma, alpha, mu_g, sg_g), A6,
                bounds_error=False,
                fill_value=np.nan,
            )
        elif Agr.ndim == 4:
            # 4D table with (mu, beta, sigma, alpha)
            beta = np.asarray(_read_first_existing(f, [
                "grids/beta_DM_grid",
                "/beta_dm_grid",
            ]))
            sigma = np.asarray(_read_first_existing(f, [
                "grids/sigma_DM_grid",
                "/sigma_dm_grid",
            ]))
            alpha = np.asarray(_read_first_existing(f, [
                "grids/alpha_grid",
                "/alpha_grid",
            ]))
            A4 = np.asarray(Agr)
            return RegularGridInterpolator(
                (mu, beta, sigma, alpha), A4,
                bounds_error=False,
                fill_value=np.nan,
            )
        elif Agr.ndim == 3:
            # 3D (mu, sigma, alpha)
            sigma = np.asarray(_read_first_existing(f, [
                "grids/sigma_DM_grid",
                "/sigma_dm_grid",
            ]))
            alpha = np.asarray(_read_first_existing(f, [
                "grids/alpha_grid",
                "/alpha_grid",
            ]))
            A = np.asarray(Agr)
            return _build_interpolator_from_arrays(mu, sigma, alpha, A)
        else:
            # 1D legacy table: A(mu)
            A1d = np.asarray(Agr)
            return _OneD_AetaInterpolator(mu, A1d)


__all__ = ["cached_A_interp"]
