"""Run MCMC sampling for the lens population parameters.

This module provides a thin wrapper around :mod:`emcee` that ties together the
mock-data generation, grid tabulation and likelihood evaluation.

Usage is intentionally simple: supply the mock observed data and a grid in
``logM_halo`` on which the lensing quantities were pre-computed.  The returned
sampler object from :mod:`emcee` can then be further analysed.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import emcee
import numpy as np
from emcee.backends import HDFBackend

from .likelihood import log_posterior
from . import likelihood as lh


def run_mcmc(
    grids,
    *,
    nwalkers: int = 50,
    nsteps: int = 3000,
    initial_guess: np.ndarray | None = None,
    backend_file: str = "chains_eta.h5",
    parallel: bool = False,
    nproc: int | None = None,
    eta: bool = True,
) -> emcee.EnsembleSampler:
    """Sample the posterior using :mod:`emcee` for 6D hyper-parameters.

    Parameters
    ----------
    grids:
        List of :class:`~make_tabulate.LensGrid` produced by
        :func:`make_tabulate.tabulate_likelihood_grids`.
    logM_sps_obs:
        Array of observed stellar masses from SPS modelling (``log10`` scale).
    nwalkers, nsteps:
        MCMC configuration.
    initial_guess:
        Initial position of the walkers in parameter space. Must have length 6
        corresponding to η = ``(alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma)``.
    backend_file:
        Filename or path for the HDF5 backend.  If a relative path is
        supplied, the file will be placed inside the ``chains`` directory.  The
        file (and directory) are created automatically if missing.
    parallel, nproc:
        If ``parallel`` is ``True`` the likelihood evaluation is distributed
        across ``nproc`` processes using :class:`multiprocessing.Pool`.  If
        ``nproc`` is ``None`` all available CPUs are used.
    """

    ndim = 6
    if initial_guess is None:
        # Default starting point (broadly near mock generator settings)
        # η = (alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma)
        initial_guess = np.array([0.1, 12.8, 2.0, 0.35, 1.05, 0.2])

    # === 使用 pathlib 构建路径 ===
    base_dir = Path(__file__).parent.resolve()
    chain_dir = base_dir / "chains"
    backend_path = chain_dir / backend_file

    # === 确保目录存在 ===
    backend_path.parent.mkdir(parents=True, exist_ok=True)
    backend = emcee.backends.HDFBackend(backend_path)
    backend.reset(nwalkers, ndim)


    print("[INFO] 从头开始采样")
    # Initialize walkers around the initial guess in 6D
    p0 = initial_guess + 0.1 * np.clip(np.random.randn(nwalkers, ndim), -1, 1) * initial_guess


    # === 运行 MCMC ===
    sampler: emcee.EnsembleSampler

    if parallel:
        if nproc is None:
            nproc = mp.cpu_count() - 2
        # Ensure A(eta) interpolator is loaded once in the parent process
        lh.init_a_interpolator()
        with mp.Pool(processes=nproc) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                log_posterior,
                args=(grids,),   # ✅ 不需要 kwargs
                backend=backend,
                pool=pool,   
            )
            sampler.run_mcmc(p0, nsteps, progress=True)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_posterior,
            args=(grids,),          # ✅ 不需要 kwargs
            backend=backend,
        )
        sampler.run_mcmc(p0, nsteps, progress=True)
    return sampler




__all__ = ["run_mcmc"]
