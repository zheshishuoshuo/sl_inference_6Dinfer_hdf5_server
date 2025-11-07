from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import h5py
import numpy as np


def _safe_attr(group: h5py.Group, key: str, value: Any) -> None:
    try:
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        if isinstance(value, (dict,)):
            group.attrs[key] = json.dumps(value, ensure_ascii=False)
        else:
            group.attrs[key] = value
    except Exception:
        group.attrs[key] = str(value)


def _copy_h5(src: h5py.Group, dst: h5py.Group) -> None:
    # copy attributes
    for k, v in src.attrs.items():
        try:
            dst.attrs[k] = v
        except Exception:
            dst.attrs[k] = str(v)
    # copy datasets and groups
    for name, item in src.items():
        if isinstance(item, h5py.Dataset):
            d = dst.create_dataset(name, data=item[...], compression="gzip")
            for k, v in item.attrs.items():
                try:
                    d.attrs[k] = v
                except Exception:
                    d.attrs[k] = str(v)
        elif isinstance(item, h5py.Group):
            g = dst.create_group(name)
            _copy_h5(item, g)


def _git_commit(root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root))
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _env_info() -> Dict[str, Any]:
    import numpy as _np
    try:
        import scipy as _scipy  # type: ignore
        scipy_v = _scipy.__version__
    except Exception:
        scipy_v = "not-installed"
    try:
        import emcee as _emcee  # type: ignore
        emcee_v = _emcee.__version__
    except Exception:
        emcee_v = "not-installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _np.__version__,
        "scipy": scipy_v,
        "emcee": emcee_v,
    }


def write_run_hdf5(
    out_path: Path | str,
    *,
    # metadata
    sample_number: int,
    lens_number: int,
    chain_length: int,
    scatter_mag: float,
    scatter_star: float,
    n_galaxy: int | None = None,
    eta: bool = True,
    true_values: Sequence[float] | None,
    seed: int | None,
    git_root: Path,
    # inputs/outputs
    samples_dict: Dict[str, np.ndarray] | None,
    zl: float | None,
    zs: float | None,
    lens_table_df,  # pandas DataFrame
    observed_table_df,  # pandas DataFrame
    grids: Sequence[Any],  # LensGrid list
    emcee_backend_path: Path | str | None,
    exports: Dict[str, np.ndarray] | None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        # metadata
        gmeta = f.create_group("metadata")
        _safe_attr(gmeta, "sample_number", int(sample_number))
        _safe_attr(gmeta, "lens_number", int(lens_number))
        _safe_attr(gmeta, "chain_length", int(chain_length))
        _safe_attr(gmeta, "scatter_mag", float(scatter_mag))
        _safe_attr(gmeta, "scatter_star", float(scatter_star))

        if n_galaxy is not None:
            _safe_attr(gmeta, "n_galaxy", int(n_galaxy))

        if eta is not None:
            _safe_attr(gmeta, "eta", bool(eta))

        if true_values is not None:
            gmeta.create_dataset("true_values", data=np.array(true_values, dtype=float))
        if seed is not None:
            _safe_attr(gmeta, "seed", int(seed))
        _safe_attr(gmeta, "git_commit", _git_commit(git_root))
        _safe_attr(gmeta, "date", _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"))
        _safe_attr(gmeta, "env_info", _env_info())

        # samples
        if samples_dict is not None:
            gs = f.create_group("samples")
            for k, v in samples_dict.items():
                gs.create_dataset(k, data=np.asarray(v), compression="gzip")
            if zl is not None:
                _safe_attr(gs, "zl", float(zl))
            if zs is not None:
                _safe_attr(gs, "zs", float(zs))

        # lens table (structured array)
        glens = f.create_group("lens")
        if lens_table_df is not None:
            # ensure columns order per spec and map mu names
            df = lens_table_df.copy()
            # provide muA/muB columns from magnification if needed
            if "muA" not in df.columns and "magnificationA" in df.columns:
                df["muA"] = df["magnificationA"]
            if "muB" not in df.columns and "magnificationB" in df.columns:
                df["muB"] = df["magnificationB"]
            cols = [
                "xA",
                "xB",
                "logM_star",
                "logM_halo",
                "logRe",
                "m_s",
                "ycaustic_kpc",
                "ycaustic_arcsec",
                "kappaA",
                "kappaB",
                "gammaA",
                "gammaB",
                "muA",
                "muB",
                "sA",
                "sB",
                "lens_id",
                "is_lensed",
            ]
            cols_present = [c for c in cols if c in df.columns]
            # build structured array with desired field names
            rec = df[cols_present].to_records(index=False)
            glens.create_dataset("table", data=rec, compression="gzip")

        # observed table
        gob = f.create_group("observed")
        if observed_table_df is not None:
            ocols = [
                "xA",
                "xB",
                "logM_star_sps_observed",
                "logRe",
                "magnitude_observedA",
                "magnitude_observedB",
            ]
            rec = observed_table_df[ocols].to_records(index=False)
            gob.create_dataset("table", data=rec, compression="gzip")

        # kernel/grids export: prefer new 2D kernel if available
        if grids is not None and len(grids) > 0 and hasattr(grids[0], "logMh_axis") and hasattr(grids[0], "gamma_h_axis"):
            # New 2D kernel layout
            first = grids[0]
            logMh_axis = np.asarray(getattr(first, "logMh_axis"))
            gamma_axis = np.asarray(getattr(first, "gamma_h_axis"))
            n_lens = len(grids)
            n_m = int(logMh_axis.size)
            n_g = int(gamma_axis.size)
            gker = f.create_group("kernel")
            gker.create_dataset("logMh_axis", data=logMh_axis)
            gker.create_dataset("gamma_h_axis", data=gamma_axis)
            ds_logM = gker.create_dataset("logM_star_true", shape=(n_lens, n_m, n_g), dtype="f8", compression="gzip")
            ds_muA = gker.create_dataset("muA", shape=(n_lens, n_m, n_g), dtype="f8", compression="gzip")
            ds_muB = gker.create_dataset("muB", shape=(n_lens, n_m, n_g), dtype="f8", compression="gzip")
            ds_msps = gker.create_dataset("logM_star_sps_obs", shape=(n_lens,), dtype="f8")
            ds_m1 = gker.create_dataset("m1_obs", shape=(n_lens,), dtype="f8")
            ds_m2 = gker.create_dataset("m2_obs", shape=(n_lens,), dtype="f8")
            ds_zl = gker.create_dataset("zl", shape=(n_lens,), dtype="f8")
            ds_zs = gker.create_dataset("zs", shape=(n_lens,), dtype="f8")
            ds_logRe = gker.create_dataset("logRe", shape=(n_lens,), dtype="f8")
            # Fill per lens without accumulating large arrays
            for i, grid in enumerate(grids):
                ds_logM[i, :, :] = np.asarray(grid.logM_star_true)
                ds_muA[i, :, :] = np.asarray(grid.muA)
                ds_muB[i, :, :] = np.asarray(grid.muB)
                ds_msps[i] = float(getattr(grid, "logM_star_sps_obs", np.nan))
                ds_m1[i] = float(getattr(grid, "m1_obs", np.nan))
                ds_m2[i] = float(getattr(grid, "m2_obs", np.nan))
                ds_zl[i] = float(getattr(grid, "zl", np.nan))
                ds_zs[i] = float(getattr(grid, "zs", np.nan))
                ds_logRe[i] = float(getattr(grid, "logRe", np.nan))
        else:
            # Legacy 1D grids (kept for backward compatibility)
            ggrids = f.create_group("grids")
            for i, grid in enumerate(grids or []):
                g = ggrids.create_group(f"lens_{i+1:05d}")
                g.create_dataset("logMh_grid", data=np.asarray(grid.logMh_grid), compression="gzip")
                g.create_dataset("logM_star", data=np.asarray(grid.logM_star), compression="gzip")
                g.create_dataset("sample_factor", data=np.asarray(grid.sample_factor), compression="gzip")
                g.create_dataset("logRe", data=np.asarray(grid.logRe))
                # optional extra fields if present
                for name in ("detJ", "beta_unit", "ycaustic"):
                    if hasattr(grid, name):
                        arr = getattr(grid, name)
                        g.create_dataset(name, data=np.asarray(arr), compression="gzip")

        # copy emcee backend
        if emcee_backend_path is not None and Path(emcee_backend_path).exists():
            with h5py.File(emcee_backend_path, "r") as src:
                gchains = f.create_group("chains")
                _copy_h5(src, gchains)

        # exports
        if exports:
            gexp = f.create_group("exports")
            for k, v in exports.items():
                gexp.create_dataset(k, data=np.asarray(v), compression="gzip")

    return out_path


def write_A_eta_hdf5(
    out_path: Path | str,
    *,
    mu_DM_grid: np.ndarray,
    sigma_DM_grid: np.ndarray,
    alpha_grid: np.ndarray,
    A_grid: np.ndarray,
    scatter_mag: float,
    scatter_star: float,
    m_lim: float,
    alpha_s: float,
    m_s_star: float,
    cache_samples: Dict[str, np.ndarray] | None = None,
    n_samples: int | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        gmeta = f.create_group("metadata")
        _safe_attr(gmeta, "scatter_mag", float(scatter_mag))
        _safe_attr(gmeta, "scatter_star", float(scatter_star))
        _safe_attr(gmeta, "m_lim", float(m_lim))
        _safe_attr(gmeta, "alpha_s", float(alpha_s))
        _safe_attr(gmeta, "m_s_star", float(m_s_star))
        _safe_attr(gmeta, "date", _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"))
        if n_samples is not None:
            _safe_attr(gmeta, "n_samples", int(n_samples))

        # Grid metadata: store ranges and lengths, not full arrays
        _safe_attr(gmeta, "mu_DM_min", float(np.min(mu_DM_grid)))
        _safe_attr(gmeta, "mu_DM_max", float(np.max(mu_DM_grid)))
        _safe_attr(gmeta, "mu_DM_len", int(np.size(mu_DM_grid)))
        _safe_attr(gmeta, "sigma_DM_min", float(np.min(sigma_DM_grid)))
        _safe_attr(gmeta, "sigma_DM_max", float(np.max(sigma_DM_grid)))
        _safe_attr(gmeta, "sigma_DM_len", int(np.size(sigma_DM_grid)))
        _safe_attr(gmeta, "alpha_min", float(np.min(alpha_grid)))
        _safe_attr(gmeta, "alpha_max", float(np.max(alpha_grid)))
        _safe_attr(gmeta, "alpha_len", int(np.size(alpha_grid)))

        ggrids = f.create_group("grids")
        ggrids.create_dataset("mu_DM_grid", data=np.asarray(mu_DM_grid))
        ggrids.create_dataset("sigma_DM_grid", data=np.asarray(sigma_DM_grid))
        ggrids.create_dataset("alpha_grid", data=np.asarray(alpha_grid))
        ggrids.create_dataset("A_grid", data=np.asarray(A_grid), compression="gzip")

        if cache_samples is not None:
            gcache = f.create_group("cache")
            for k, v in cache_samples.items():
                arr = np.asarray(v)
                # Create dataset first, then write in slices to reduce peak memory
                dset = gcache.create_dataset(
                    k,
                    shape=arr.shape,
                    dtype=arr.dtype,
                    chunks=True,
                    compression="gzip",
                )
                if arr.ndim == 1:
                    # Heuristic chunk size: ~1e6 bytes per write
                    step = max(1, 1_000_000 // max(1, arr.itemsize))
                    for s in range(0, arr.shape[0], step):
                        e = min(s + step, arr.shape[0])
                        dset[s:e] = arr[s:e]
                else:
                    dset[...] = arr
    return out_path


__all__ = ["write_run_hdf5", "write_A_eta_hdf5"]


def write_A_eta_hdf5_1d(
    out_path: Path | str,
    *,
    mu_DM_grid: np.ndarray,
    A_grid: np.ndarray,
    scatter_mag: float,
    scatter_star: float,
    m_lim: float,
    alpha_s: float,
    m_s_star: float,
    cache_samples: Dict[str, np.ndarray] | None = None,
    n_samples: int | None = None,
) -> Path:
    """Write a 1D A(eta) table over mu_DM only.

    Stores minimal metadata plus a ``grids`` group with ``mu_DM_grid`` and
    1D ``A_grid``. Optionally stores cached Monte Carlo samples under
    ``/cache`` for reproducibility.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        gmeta = f.create_group("metadata")
        _safe_attr(gmeta, "scatter_mag", float(scatter_mag))
        _safe_attr(gmeta, "scatter_star", float(scatter_star))
        _safe_attr(gmeta, "m_lim", float(m_lim))
        _safe_attr(gmeta, "alpha_s", float(alpha_s))
        _safe_attr(gmeta, "m_s_star", float(m_s_star))
        _safe_attr(gmeta, "date", _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"))
        if n_samples is not None:
            _safe_attr(gmeta, "n_samples", int(n_samples))

        # Grid metadata: store ranges and length
        _safe_attr(gmeta, "mu_DM_min", float(np.min(mu_DM_grid)))
        _safe_attr(gmeta, "mu_DM_max", float(np.max(mu_DM_grid)))
        _safe_attr(gmeta, "mu_DM_len", int(np.size(mu_DM_grid)))

        ggrids = f.create_group("grids")
        ggrids.create_dataset("mu_DM_grid", data=np.asarray(mu_DM_grid))
        ggrids.create_dataset("A_grid", data=np.asarray(A_grid), compression="gzip")

        if cache_samples is not None:
            gcache = f.create_group("cache")
            for k, v in cache_samples.items():
                gcache.create_dataset(k, data=np.asarray(v), compression="gzip")
    return out_path


__all__.extend(["write_A_eta_hdf5_1d"])
