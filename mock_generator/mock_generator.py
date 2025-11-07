import numpy as np
import random
import pandas as pd
from .lens_properties import observed_data, empty_lens_data
from .lens_model import LensModel, kpc_to_arcsec
from ..sl_cosmology import Dang
from tqdm import tqdm
from .mass_sampler import generate_samples, sample_m_s
from pathlib import Path
import datetime as _dt
import gc
try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None  # type: ignore

# SPS PARAMETER
# M_star = alpha_sps * M_sps
# logM_star = log_alpha_sps + logM_sps

import multiprocessing

# Per-process RNG for multiprocessing workers
_LOCAL_RNG: np.random.Generator | None = None


def _init_worker_seed(base_seed: int | None) -> None:
    """Top-level initializer for multiprocessing workers.

    Sets up a per-process NumPy Generator and also seeds numpy/random.
    Placed at module scope so it is picklable by multiprocessing on
    spawn-based platforms (e.g., macOS, Windows, Python 3.12+).

    Parameters
    ----------
    base_seed : int | None
        If provided, every worker uses the same seed value to ensure
        identical initialisation across processes. If None, a default
        Generator is created without explicit seeding.
    """
    global _LOCAL_RNG
    if base_seed is None:
        _LOCAL_RNG = np.random.default_rng()
        return
    s = int(base_seed)
    np.random.seed(s)
    _LOCAL_RNG = np.random.default_rng(s)
    random.seed(s)


def _sim_star(args):
    return simulate_single_lens(*args)


def simulate_single_lens(i, samples, logalpha_sps_sample,
                        maximum_magnitude, zl, zs, nbkg,
                        alpha_s, m_s_star,
                        rng: np.random.Generator | None = None,
                        **kwargs):
    if 'gamma_dm' in kwargs:
        raise ValueError("gamma_dm is no longer supported; use gamma_in.")
    """Simulate all sources for a single lens.

    Returns
    -------
    tuple[dict, int]
        A tuple containing (result, i):
        - result: a single record (dict) per galaxy. Contains lensing
          outputs if lensed; otherwise a placeholder with metadata and
          is_lensed=False.
        - i: the lens index (echoed back for aggregation in parallel mode)
    """
    logM_star_sps = samples['logM_star_sps'][i]
    logM_star = logM_star_sps + logalpha_sps_sample[i]
    logM_halo = samples['logM_halo'][i]
    logRe = samples['logRe'][i]
    # Required halo concentration parameter
    c_halo = samples['c_halo'][i] if 'c_halo' in samples else None
    gamma_in = samples['gamma_in'][i]
    model = LensModel(logM_star=logM_star, logM_halo=logM_halo,
                      logRe=logRe, zl=zl, zs=zs, gamma_in=gamma_in, c_halo=c_halo)
    # Precompute halo-level quantities used across candidates
    ycaust_kpc = model.solve_ycaustic()
    logMh5 = model.logMh5()
    model._ycaust_kpc = ycaust_kpc  # cache for downstream usage if needed
    if ycaust_kpc is None:
        # No caustic => still return a record to keep alignment
        result = empty_lens_data(
            model,
            logM_star_sps=logM_star_sps,
            logM_star=logM_star,
            logM_halo=logM_halo,
            logRe=logRe,
            maximum_magnitude=maximum_magnitude,
            logalpha_sps=logalpha_sps_sample[i],
            zl=zl,
            zs=zs,
            gamma_in=gamma_in,
            c_halo=c_halo,
            beta_unit=None,
            ycaustic_kpc=ycaust_kpc,
        )
        result['lens_id'] = i
        result['is_lens'] = False
        result['logMh5'] = logMh5

        return result, i
        

    # Choose RNG: prefer explicit one; fall back to per-process RNG
    local_rng = rng if rng is not None else _LOCAL_RNG
    if local_rng is None:
        local_rng = np.random.default_rng()
    
    lambda_i = np.pi * ycaust_kpc**2 * nbkg
    N_i = local_rng.poisson(lambda_i)

    # if N_i != 0:
    #     # return []
    #     print(f"Lens {i}: ycaustic = {ycaust_kpc:.2f} kpc, N_sources = {N_i}")

    results = []
    found_lens = False

    for _ in range(N_i):
        beta_unit = np.sqrt(local_rng.random())
        m_s = sample_m_s(alpha_s, m_s_star, rng=local_rng)
        # Stage A: fast geometric rejection using precomputed caustic size
        if (ycaust_kpc is None) or (not np.isfinite(ycaust_kpc)) or (ycaust_kpc <= 0):
            # Should be handled above, but keep as guard
            continue
        beta_kpc = float(beta_unit) * float(ycaust_kpc)
        if not (beta_kpc < ycaust_kpc):
            # Outside caustic: skip heavy computation for this candidate
            continue
        result = observed_data(
            model,
            beta_unit=beta_unit,
            m_s=m_s,
            maximum_magnitude=maximum_magnitude,
            logalpha_sps=logalpha_sps_sample[i],
            logM_star=logM_star,
            logM_star_sps=logM_star_sps,
            logM_halo=logM_halo,
            logRe=logRe,
            zl=zl,
            zs=zs,
            gamma_in=gamma_in,
            c_halo=c_halo,
            caustic=False,
        )
        if result.get('is_lensed', False):
            result['lens_id'] = i
            result['ycaustic_kpc'] = ycaust_kpc
            result['ycaustic_arcsec'] = kpc_to_arcsec(ycaust_kpc, zl, Dang)
            result['logMh5'] = logMh5
            results.append(result)
            found_lens = True
            break

    if not found_lens:
        # return a record even if not a lens using helper
        result = empty_lens_data(
            model,
            logM_star_sps=logM_star_sps,
            logM_star=logM_star,
            logM_halo=logM_halo,
            logRe=logRe,
            maximum_magnitude=maximum_magnitude,
            logalpha_sps=logalpha_sps_sample[i],
            zl=zl,
            zs=zs,
            gamma_in=gamma_in,
            c_halo=c_halo,
            beta_unit=None,
            ycaustic_kpc=ycaust_kpc,
        )
        result['lens_id'] = i
        result['is_lens'] = False
        result['logMh5'] = logMh5
        return result, i
    else:
        # return the lensed system (first found)
        result = results[0]
        result['is_lens'] = True
        result['logMh5'] = logMh5
        return result, i


def run_mock_simulation(
    n_samples,
    maximum_magnitude=26.5,
    zl=0.3,
    zs=2.0,
    if_source=False,
    process=None,
    alpha_s=-1.3,
    m_s_star=24.5,
    logalpha: float = 0,
    seed = None,
    deterministic: bool = False,
    nbkg: float = 4e-4,
    *,
    cache_every: int | None = None,
    cache_dir: str | Path | None = None,
):
    """Run a mock strong-lens simulation.

    Parameters
    ----------
    n_samples : int
        Number of lens galaxies to simulate.
    nbkg : float, optional
        Surface density of background sources in ``kpc^-2``.
    """

    # Determine effective seed; create a single unified Generator for NumPy
    effective_seed = 12345 if deterministic else seed
    rng = np.random.default_rng(int(effective_seed)) if effective_seed is not None else np.random.default_rng()
    # Keep Python's builtin random in sync for any incidental usage elsewhere
    if effective_seed is not None:
        random.seed(int(effective_seed))

    logalpha_sps_sample = np.full(n_samples, logalpha)
    # Use the unified RNG for generating base samples
    samples = generate_samples(n_samples, rng=rng)

    # gamma_dm removed: only gamma_in is used/propagated

    # Optional HDF5 chunk cache to control memory
    use_cache = bool(cache_every) and (h5py is not None)
    cache_path: Path | None = None
    written_chunks = 0
    buffer: list[dict] = []
    if use_cache:
        base_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[1] / "chains"
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            base_dir = Path.cwd()
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cache_path = base_dir / f"simulate_cache_{ts}.h5"
        # init file with root groups
        try:
            with h5py.File(cache_path, "w") as f:
                f.create_group("chunks")
        except Exception:
            cache_path = None
            use_cache = False

    def _flush_buffer() -> None:
        nonlocal written_chunks, buffer
        if not use_cache or not buffer:
            return
        assert cache_path is not None
        df_chunk = pd.DataFrame(buffer)
        with h5py.File(cache_path, "a") as f:
            g = f["chunks"].create_group(f"chunk_{written_chunks:05d}")
            rec = df_chunk.to_records(index=False)
            g.create_dataset("table", data=rec, compression="gzip")
        written_chunks += 1
        buffer = []
        gc.collect()

    # Prepare per-lens arrays to record the sampled source that produced a lens
    m_s_array = np.full(n_samples, np.nan)
    beta_unit_array = np.full(n_samples, np.nan)

    if process is None or process == 0:
        lens_results = [] if not use_cache else None
        columns = None  # columnar accumulation when not caching
        for i in tqdm(range(n_samples), desc="Processing lenses"):
            result, idx = simulate_single_lens(
                i, samples, logalpha_sps_sample, maximum_magnitude, zl, zs, nbkg, alpha_s, m_s_star, rng,
            )
            if use_cache:
                buffer.append(result)
                if cache_every and len(buffer) >= int(cache_every):
                    _flush_buffer()
            else:
                if columns is None:
                    # Initialize columnar buffers with first row
                    columns = {k: [v] for k, v in result.items()}
                else:
                    # Determine current row count from any existing column
                    row_count = len(next(iter(columns.values()))) if columns else 0
                    # Add any new keys introduced by this result
                    for k in result.keys():
                        if k not in columns:
                            columns[k] = [None] * row_count
                    # Append values for all known columns (missing keys get None)
                    for k in list(columns.keys()):
                        columns[k].append(result.get(k, None))
            # Record m_s and beta_unit for the lensed source if present
            m_s_val = result.get('m_s', None)
            beta_val = result.get('beta_unit', None)
            m_s_array[idx] = np.nan if m_s_val is None else m_s_val
            beta_unit_array[idx] = np.nan if beta_val is None else beta_val
    else:
        # Multiprocessing path: use a global initializer to avoid pickling issues
        args = [
            (i, samples, logalpha_sps_sample, maximum_magnitude, zl, zs, nbkg,
             alpha_s, m_s_star, None)
            for i in range(n_samples)
        ]
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(
            process,
            initializer=_init_worker_seed,
            initargs=(int(effective_seed) if effective_seed is not None else None,)
        )
        with pool:
            it = pool.imap_unordered(_sim_star, args, chunksize=128)
            pbar = tqdm(total=n_samples, desc=f"Processing lenses (process={process})")
            if use_cache:
                lens_results = None
                for sub in it:
                    if sub:
                        result, idx = sub
                        buffer.append(result)
                        if cache_every and len(buffer) >= int(cache_every):
                            _flush_buffer()
                        m_s_val = result.get('m_s', None)
                        beta_val = result.get('beta_unit', None)
                        m_s_array[idx] = np.nan if m_s_val is None else m_s_val
                        beta_unit_array[idx] = np.nan if beta_val is None else beta_val
                    pbar.update(1)
                _flush_buffer()
            else:
                lens_results = []
                for sub in it:
                    if sub:
                        result, idx = sub
                        lens_results.append(result)
                        m_s_val = result.get('m_s', None)
                        beta_val = result.get('beta_unit', None)
                        m_s_array[idx] = np.nan if m_s_val is None else m_s_val
                        beta_unit_array[idx] = np.nan if beta_val is None else beta_val
                    pbar.update(1)
            pbar.close()

    # Build final DataFrame either from memory or by reading cache
    if use_cache and cache_path is not None:
        rows: list[pd.DataFrame] = []
        try:
            with h5py.File(cache_path, "r") as f:
                ch = f["chunks"]
                for name in ch.keys():
                    rec = ch[name]["table"][...]
                    rows.append(pd.DataFrame.from_records(rec))
            df_lens = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        finally:
            try:
                Path(cache_path).unlink()
            except Exception:
                pass
    else:
        if lens_results and isinstance(lens_results, list) and len(lens_results) > 0 and isinstance(lens_results[0], dict):
            df_lens = pd.DataFrame(lens_results)
        else:
            df_lens = pd.DataFrame(columns or {})
    if df_lens.empty:
        mock_lens_data = pd.DataFrame(columns=df_lens.columns)
        mock_observed_data = pd.DataFrame(columns=[
            'xA', 'xB', 'logM_star_sps_observed', 'logRe',
            'magnitude_observedA', 'magnitude_observedB'
        ])
    else:
        mock_lens_data = df_lens[df_lens['is_lensed']].copy()
        mock_observed_data = mock_lens_data[
            ['xA', 'xB', 'logM_star_sps_observed', 'logRe',
             'magnitude_observedA', 'magnitude_observedB']
        ].copy()

    # Augment returned samples with m_s and beta_unit arrays
    samples = dict(samples)
    samples['m_s'] = m_s_array
    samples['beta_unit'] = beta_unit_array

    if if_source:
        # Return full results including the underlying sampled inputs for HDF5 export
        return df_lens, mock_lens_data, mock_observed_data, samples
    else:
        return mock_lens_data, mock_observed_data

if __name__ == "__main__":
        # 串行
    mock_lens_data, mock_observed_data = run_mock_simulation(1000, process=0)

    # 默认行为（串行）
    mock_lens_data, mock_observed_data = run_mock_simulation(1000)

    # 并行，使用 8 核
    mock_lens_data, mock_observed_data = run_mock_simulation(1000, process=8)

    df_lens, mock_lens_data, mock_observed_data, samples = run_mock_simulation(1000, process=8, if_source=True)
