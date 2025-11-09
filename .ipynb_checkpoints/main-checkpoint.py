from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import argparse
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib

from .mock_generator.mock_generator import run_mock_simulation
from .mock_generator.mass_sampler import MODEL_PARAMS
from .make_tabulate import tabulate_likelihood_grids
from .run_mcmc import run_mcmc
from .config import SCATTER
from .hdf5_io import write_run_hdf5
from . import likelihood as likelihood_mod


# ---------------------------------------------------------------------------
# External 2D dark matter parameter grid (logMh, gamma_h)
# - Created once at import time and reused.
# - Default ranges: logMh in [11.0, 15.0], gamma_h in [0.2, 2.2].
# - Kept external to the inference pipeline; not re-generated inside functions.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DMGrid2D:
    logMh: np.ndarray
    gamma_h: np.ndarray


def build_dm_grid2d(
    *,
    logMh_min: float = 11.0,
    logMh_max: float = 15.0,
    n_logMh: int = 30,
    gamma_min: float = 0,
    gamma_max: float = 2,
    n_gamma: int = 30,
) -> DMGrid2D:
    """Construct a reusable 2D grid over (logMh, gamma_h).

    This defines only the parameter axes. No likelihood/tabulation is done here.
    """
    logMh_axis = np.linspace(float(logMh_min), float(logMh_max), int(n_logMh))
    gamma_axis = np.linspace(float(gamma_min), float(gamma_max), int(n_gamma))
    return DMGrid2D(logMh=logMh_axis, gamma_h=gamma_axis)


# Create once and keep module-global for reuse
DM_GRID_2D: DMGrid2D = build_dm_grid2d()


def _configure_matplotlib_backend() -> None:
    """Choose a safe backend for Linux/headless environments.

    - Respect `MPLBACKEND` if set by user.
    - On Linux without DISPLAY, switch to non-interactive 'Agg'.
    - Otherwise leave Matplotlib defaults (user/site config) intact.
    """
    if os.environ.get("MPLBACKEND"):
        # User explicitly set backend, do not override
        return
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            # Fallback: ignore if backend switch fails; plotting will be skipped
            pass


_configure_matplotlib_backend()
import matplotlib.pyplot as plt

def main(
    interact: bool = False,
    *,
    n_galaxy: int | None = None,
    save_plots: bool | None = None,
    eta: bool = True,
    dm_grid: DMGrid2D | None = None,
) -> None:
    # os.remove(os.path.join(os.path.dirname(__file__), "chains", "chains.h5")) if os.path.exists(os.path.join(os.path.dirname(__file__), "chains", "chains.h5")) else None
    # Generate mock data for samples with fixed logalpha
    # mock_lens_data, mock_observed_data = run_mock_simulation(300, logalpha=0.1)
    logalpha = 0.0
    model_p = MODEL_PARAMS["deVauc"]
    # Whether to use A(eta) correction (can override via CLI)

    # Initialise A(eta) interpolator (interactive option)
    # try:
    #     likelihood_mod.init_a_interpolator(interactive=interact)
    # except Exception as e:
    #     print(f"[WARN] Failed to initialise A(eta) interpolator: {e}")

    # Prepare external DM 2D grid (kept external/global)
    dm_grid = DM_GRID_2D if dm_grid is None else dm_grid

    # Generate a reasonably sized mock sample so that running this script is
    # fast but still demonstrates the full workflow.
    # seed = int(np.random.randint(0, 10000))
    seed = 420

    # default sample size (can override via CLI)
    n_galaxy = int(n_galaxy) if n_galaxy is not None else 30000

    print(f"Generating mock data with {n_galaxy} galaxies, logalpha={logalpha}, seed={seed} ...")
     # 4e-4 for 200k
    df_lens, mock_lens_data, mock_observed_data, samples_dict = run_mock_simulation(
        n_galaxy,
          logalpha=logalpha, seed=seed, nbkg=4e-4, if_source=True
    )
    # print(np.mean(mock_lens_data["logM_halo"].values))
    # print(mock_lens_data.shape)
    # logM_sps_obs = mock_observed_data["logM_star_sps_observed"].values

    # mock_lens_data.to_csv("mock_lens_data.csv", index=False)

    # Precompute 2D kernel tables (hyperparameter-free) on external DM grid
    grids = tabulate_likelihood_grids(
        mock_observed_data,
        dm_grid,
        # n_jobs=max(1, mp.cpu_count() // 2),
        n_jobs=None,
    )

    nsteps = 3000

    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backend_file = f"chains_{int(mock_lens_data.shape[0])}lens_noeta_{ts}_{seed}.h5"


    # Run MCMC with full 2D-kernel likelihood
    sampler = run_mcmc(
        grids,
        nsteps=nsteps,
        nwalkers=60,
        # 6D initial guess: (alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma)
        initial_guess=np.array([0.01, 12.91, 2.04, 0.37, 1.0, 0.2]),
        backend_file=backend_file,
        parallel=True,
        nproc=max(1, mp.cpu_count() - 2),
        eta=eta,
    )
    burnin = 0
    chain = sampler.get_chain(discard=burnin, flat=True)
    print("MCMC sampling completed.")

    samples = chain.reshape(-1, chain.shape[-1])

    # 转为 DataFrame 并加上列名（6D 参数）
    param_names = [
        r"$\alpha_{\rm sps}$",
        r"$\mu_{h}$",
        r"$\beta_{h}$",
        r"$\sigma_{h}$",
        r"$\mu_{\gamma}$",
        r"$\sigma_{\gamma}$",
    ]

    df_samples = pd.DataFrame(samples, columns=param_names)

    # # 画 pairplot
    # sns.pairplot(
    #     df_samples,
    #     diag_kind="kde",
    #     markers=".",
    #     plot_kws={"alpha": 0.5, "s": 10},
    #     corner=True
    # )

    # 真值
    # Ground-truth for reference: η = (alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma)
    true_values = [logalpha, model_p["mu_h0"], model_p["beta_h"], model_p["sigma_h"], 1.0, 0.2]

    # 绘制与保存/展示图像（Linux 上默认保存而非显示）
    if save_plots is None:
        # If using non-interactive backend, default to saving
        non_interactive = plt.get_backend().lower() in {"agg", "pdf", "svg"}
        save_plots = non_interactive

    plot_dir = Path(__file__).resolve().parent / "chains" / "plots"
    if save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    if df_samples.shape[1] == 1:
        ax = df_samples.iloc[:, 0].plot(kind="kde")
        ax.axvline(true_values[0], color="red", linestyle="--", linewidth=1)
        ax.set_xlabel(param_names[0])
        if save_plots:
            out = plot_dir / f"kde_alpha_sps_{ts}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {out}")
        else:
            plt.show()
    else:
        # Matplotlib-only corner plot (no seaborn)
        npar = df_samples.shape[1]
        fig, axes = plt.subplots(npar, npar, figsize=(3.2 * npar, 3.2 * npar))
        if npar == 1:
            axes = np.array([[axes]])
        for i in range(npar):
            for j in range(npar):
                ax = axes[i, j]
                if i < j:
                    ax.axis("off")
                    continue
                xi = df_samples.iloc[:, j]
                yi = df_samples.iloc[:, i]
                if i == j:
                    ax.hist(xi, bins=60, density=True, color="#4c72b0", alpha=0.6)
                    ax.axvline(true_values[i], color="red", linestyle="--", linewidth=1)
                    ax.set_xlabel(str(df_samples.columns[j]))
                else:
                    ax.scatter(xi, yi, s=5, alpha=0.1, color="#4c72b0")
                    ax.axvline(true_values[j], color="red", linestyle="--", linewidth=0.8)
                    ax.axhline(true_values[i], color="red", linestyle="--", linewidth=0.8)
                # Ticks / labels: only left column and bottom row
                if j == 0:
                    ax.set_ylabel(str(df_samples.columns[i]))
                else:
                    ax.set_yticklabels([])
                if i == npar - 1:
                    ax.set_xlabel(str(df_samples.columns[j]))
                else:
                    ax.set_xticklabels([])
        fig.tight_layout()
        if save_plots:
            out = plot_dir / f"pairplot_{ts}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {out}")
        else:
            plt.show()


    print("Finished MCMC. Chain shape:", chain.shape)

    # ---- Write HDF5 run file ----
    try:
        base_dir = Path(__file__).parent.resolve()
        chains_dir = base_dir / "chains"
        chains_dir.mkdir(exist_ok=True)
        # Use the same timestamp as the MCMC backend file to keep them aligned
        run_filename = chains_dir / f"chains_{mock_lens_data.shape[0]}lens_{ts}.h5"

        # Exports
        log_prob_flat = sampler.get_log_prob(discard=burnin, flat=True)
        acceptance_frac = sampler.acceptance_fraction
        exports = {
            "samples_flat": chain,
            "log_prob": log_prob_flat,
            "acceptance_frac": acceptance_frac,
        }

        # Determine emcee backend absolute path
        emcee_backend_path = base_dir / "chains" / backend_file

        write_run_hdf5(
            run_filename,
            sample_number=int(samples_dict["logM_star_sps"].size),
            lens_number=int(mock_lens_data.shape[0]),
            chain_length=int(nsteps),
            scatter_mag=float(SCATTER.mag),
            scatter_star=float(SCATTER.star),
            n_galaxy=n_galaxy,
            eta=eta,
            # Save true values for 6D parameters (η)
            true_values=[logalpha, model_p["mu_h0"], model_p["beta_h"], model_p["sigma_h"], 1.0, 0.2],
            seed=seed,
            git_root=base_dir,
            samples_dict={
                "logM_star_sps": samples_dict["logM_star_sps"],
                "logRe": samples_dict["logRe"],
                "logM_halo": samples_dict["logM_halo"],
            },
            zl=0.3,
            zs=2.0,
            lens_table_df=df_lens,
            observed_table_df=mock_observed_data,
            grids=grids,
            emcee_backend_path=emcee_backend_path,
            exports=exports,
        )
        print(f"Saved run results to: {run_filename}")
    except Exception as e:
        print(f"[WARN] Failed to write HDF5 run file: {e}")
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SL inference pipeline")
    parser.add_argument("--interact", action="store_true", help="Interactively select A(eta) table")
    parser.add_argument("--n-galaxy", type=int, default=None, help="Number of galaxies to simulate (default 200000)")
    parser.add_argument("--save-plots", action="store_true", help="Force saving plots to files")
    parser.add_argument("--show-plots", action="store_true", help="Force showing plots interactively")
    parser.add_argument("--no-eta", action="store_true", help="Disable A(eta) correction in likelihood")
    args = parser.parse_args()
    save_plots = args.save_plots or (not args.show_plots)
    if args.show_plots:
        # If user forces interactive view, try a GUI backend if possible
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            pass
    main(interact=args.interact, n_galaxy=args.n_galaxy, save_plots=save_plots, eta=(not args.no_eta))
