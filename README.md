**项目简介**
- 本库用于强引力透镜样本的模拟、似然栅格预计算与基于 emcee 的超参数后验推断。核心流程为：用物理一致的质量-尺寸-晕质量关系生成模拟透镜 → 对每个透镜在晕质量网格上求解镜像几何与光度选择因子并做缓存 → 使用 A(η) 归一化修正与联合似然，对超参数 η 进行 MCMC 采样。

**快速开始**
- 依赖：Python 3.10+，numpy，scipy，pandas，emcee，matplotlib，tqdm，h5py。
- 运行端到端示例：
  1) 生成模拟样本与预计算栅格并运行 MCMC（保存结果和图）：
     - `main.py:1` 内置示例入口，可直接运行：`python -m sl_inference_4Dinfer_hdf5.main --n-galaxy 100000 --save-plots`
  2) 仅构建每个透镜的似然栅格：
     - 使用 `make_tabulate.py:1` 的 `tabulate_likelihood_grids`，传入观测表和 `logMh` 网格。
  3) 直接运行采样：
     - 使用 `run_mcmc.py:1` 的 `run_mcmc(grids, logM_sps_obs, ...)`。

**目录结构**
- 顶层 Python 模块与脚本：
  - `main.py:1`：端到端管线示例，串起模拟→预计算→MCMC→保存 HDF5 结果与图。
  - `likelihood.py:1`：6 维超参数 η 的先验/似然/后验与 A(η) 插值器初始化。
  - `make_tabulate.py:1`：对每个透镜在 `logMh` 网格上预计算与超参数无关的量（`LensGrid`）。
  - `run_mcmc.py:1`：emcee 采样封装（支持并行与断点续跑）。
  - `hdf5_io.py:1`：统一的 HDF5 读写：运行产物、A(η) 表等。
  - `cached_A.py:1`：从 `aeta_tables/` 读取 A(η) HDF5，并返回 1D/3D/4D 插值器。
  - `grid_generator.py:1`：在多维参数网格上批量生成透镜量并流式写 HDF5（工具脚本）。
  - `build_k_table.py:1`：构建 `K(μ1, μ2)` 插值表（已对源星等积分）。
  - `mock_tabulate_manager.py:1`：大样本批处理：生成 mock → 分批 tabulate → HDF5 缓存/子集/评估。
  - `config.py:1`：全局配置（观测散射等）。
  - `utils.py:1`：选择函数与星等似然等实用函数。
  - `sl_cosmology.py:1`：角径距、临界面密度等宇宙学量。

- 子包与数据：
  - `mock_generator/:1`：模拟器子包（见下文“模拟器子包”）。
  - `sl_profiles/:1`：质量分布与预制栅格（deV、gNFW 等）。
  - `aeta_generator/:1`：从 mock bank 与 K 表构建多维 A(η) 表的工具脚本。
  - `aeta_tables/:1`：本地 A(η) HDF5 表存放目录（`Aeta_*.h5`）。
  - `chains/:1`：MCMC 采样与中间缓存默认输出目录（代码运行后生成）。

**模块说明**
- `main.py:1`
  - 功能：端到端示例。初始化 A(η) 插值器，生成 mock 样本，构建 `LensGrid` 列表，运行 emcee 采样，出图并写入运行 HDF5。
  - 关键参数：`--n-galaxy`、`--no-eta`、`--interact`（交互选择 A(η) 表）。

- `likelihood.py:1`
  - 功能：定义 6 维超参数 η = (alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma) 的先验与联合似然；封装 `init_a_interpolator` 从 `aeta_tables/` 载入 A(η)。
  - 接口：`log_prior`、`log_likelihood`、`log_posterior`、`precompute_grids`。

- `make_tabulate.py:1`
  - 功能：对每个透镜在 `logMh` 网格上预计算与 η 无关的量，输出 `LensGrid`（含 `logM_star`、`sample_factor`、`detJ`、`beta_unit`、`ycaustic` 等）。支持分批写入临时 HDF5 缓存以降低峰值内存。
  - 接口：`LensGrid` 数据类、`tabulate_likelihood_grids(mock_observed_data, logMh_grid, ...)`。

- `run_mcmc.py:1`
  - 功能：基于 emcee 的采样器封装。支持并行、HDF5 backend 断点续跑、初值设置、控制步数/行者数量。
  - 接口：`run_mcmc(grids, logM_sps_obs, nwalkers, nsteps, ...)`。

- `hdf5_io.py:1`
  - 功能：
    - `write_run_hdf5(...)` 保存一次完整运行的元数据、样本、观测表、每透镜栅格以及 emcee 后端；
    - `write_A_eta_hdf5(...)` 与 `write_A_eta_hdf5_1d(...)` 保存 A(η) 表及元数据/缓存样本。

- `cached_A.py:1`
  - 功能：在本地 `aeta_tables/` 目录自动发现并读取 `Aeta_*.h5`；支持 1D/3D/4D（含 β 维）文件并返回相应维度的插值器。可交互选择表。

- `grid_generator.py:1`
  - 功能：命令行工具，在给定轴规格上（`logMh, logMstar, logRe, beta`）批量求解镜像、放大率、kappa/gamma、detJ、ycaustic 等并流式写入 HDF5。
  - 使用：`python grid_generator.py --axes logMh:10:16:0.1,logMstar:11:13:0.05,logRe:0:2:0.05,beta:0:1:0.01 --out bank.h5 --n-proc 8 --overwrite`。

- `build_k_table.py:1`
  - 功能：构建并保存 `K(μ1, μ2)=∫ p_det(μ1,ms)p_det(μ2,ms)p(ms)dms` 表；提供内存/文件两种路径与插值器装载器。
  - 接口：`build_K_table`、`save_K_table_hdf5`、`load_K_interpolator`。

- `mock_tabulate_manager.py:1`
  - 功能：大样本工作流工具：
    - `generate_and_save_mock_tabulate`：分批生成 mock、分批 tabulate、分批写 HDF5；
    - `load_mock_and_grids`：按全局索引加载子集；
    - `save_subset`：将子集另存；
    - `compute_posterior_from_file`：基于已存 HDF5 直接计算对数似然/后验（流式）。

- `config.py:1`
  - 功能：集中管理观测散射参数（`SCATTER.star`、`SCATTER.mag`）和常量配置。

- `utils.py:1`
  - 功能：
    - `selection_function(mu, m_lim, ms, sigma_m)`：单个像的探测选择函数；
    - `mag_likelihood(m_obs, mu, ms, sigma_m)`：星等观测似然。

- `sl_cosmology.py:1`
  - 功能：角径距 `Dang`、临界面密度 `Sigma_cr`、临界密度 `rhoc` 等基础宇宙学量。

**模拟器子包（mock_generator）**
- `mock_generator/mock_generator.py:1`
  - 功能：生成透镜样本。对每个透镜按源面背景密度抽样源，快速几何拒绝+镜像解算，输出全表与“被透镜化子集”和“观测量表”，支持分批 HDF5 缓存以控内存。
  - 返回：
    - `df_lens`（每个候选源的逐行结果，含 `is_lensed` 标记），
    - `mock_lens_data`（被透镜化子集），
    - `mock_observed_data`（推断用观测表：xA/xB、logRe、两像星等、logM★_sps_obs），
    - `samples`（生成时的 logM★_sps、logRe、logMh、γ_in、c_halo、以及产生透镜的源 m_s 与 beta_unit）。

- `mock_generator/mass_sampler.py:1`
  - 功能：基于文献参数的质量与尺寸分布采样：
    - `mstar_gene`（skew-normal 的 logM★_sps），
    - `logRe_given_logM`（尺寸-质量关系），
    - `logMh_given_logM_logRe`（晕质量条件分布），
    - `sample_gamma`（内密度坡度 γ_in）。
  - 常量：`MODEL_PARAMS`（deVauc/SerExp/Sersic 三种模型参数）。

- `mock_generator/lens_model.py:1`
  - 功能：透镜物理模型，提供 α、κ、γ、μ 及临界线/爱因斯坦半径求解；恒星分量用 deV，暗晕用 gNFW（γ_in 与 c_halo）。

- `mock_generator/lens_solver.py:1`
  - 功能：
    - `solve_single_lens(model, beta_unit)`：由源偏移求像位置；
    - `solve_lens_parameters_from_obs[_yc]`：由观测像位置/尺寸/晕质量反推 M★ 与源偏移；
    - `compute_detJ`：数值求解 (xA,xB)→(logM★,β) 的雅可比行列式。

- `mock_generator/lens_properties.py:1`
  - 功能：
    - `lens_properties`：给定 `LensModel` 与源偏移，返回像位置、放大率、κ/γ、θ_E 等；
    - `observed_data`：生成观测量（两像星等、logM★_sps_obs），并打包物理/观测属性；
    - `empty_lens_data`：未透镜化时的占位记录（保持列齐）。

**质量分布与栅格（sl_profiles）**
- `sl_profiles/deVaucouleurs.py:1`、`sl_profiles/gnfw.py:1` 等：
  - 提供 de Vaucouleurs 恒星分布与 gNFW 暗晕的投影/累计质量等快速计算；仓库内伴随 `*.hdf5` 栅格以加速评估。

**A(η) 生成工具（aeta_generator）**
- `aeta_generator/compute_Aeta_grid.py:1`
  - 功能：从 mock bank 与 `K(μ1, μ2)` 表出发，基于重要性权重在 (μ_DM, β_DM, σ_DM, α) 网格上评估 A(η)，支持分块/并行与大体量 HDF5 输出。
  - 相关：`build_k_table.py:1` 提供 `K` 表构建与插值；`aeta_tables/:1` 存放生成的 `Aeta_*.h5`。

**产出与文件**
- MCMC 与运行产物默认写入 `chains/:1`，包括：
  - emcee HDF5 backend（可续跑）、
  - 运行 HDF5（元数据、样本、观测表、每透镜栅格、导出诊断）。
- 预制/生成的数据表：
  - `aeta_tables/Aeta_*.h5:1`：A(η) 表（1D/3D/4D 兼容）。
  - `K_*.h5:1`：K 表；`p_mstar_table.h5:1`：恒星质量先验表（若使用）。

**性能与实践建议**
- 大样本建议：
  - `make_tabulate.tabulate_likelihood_grids(..., cache_every=..., cache_dir=...)` 开启分批缓存，避免内存峰值；
  - 采样时优先使用 HDF5 backend 并控制并行进程数，避免与 BLAS 线程过度竞争；
  - `mock_tabulate_manager.py:1` 提供“分批生成-分批栅格-流式评估”的一体化接口。

**引用**
- 若在学术工作中使用本库，请在文稿中引用相应的质量分布与透镜统计文献，并在方法中简述本库的 A(η) 归一化与 6 维超参数推断框架。
