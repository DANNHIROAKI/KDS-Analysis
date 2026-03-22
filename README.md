# KDTree Baseline 实验落地仓库

这个仓库把你给出的实验计划完整落地成一个**可一键执行**的实验外壳，同时严格遵守两个约束：

1. `vendor/KDTree.zip` 与其解压镜像完全按原样保留；
2. 所有实验驱动、数据生成、指标采集、分析与汇总代码都放在外层，不改动基线内部任何文件。

## 目录结构

- `vendor/KDTree.zip` / `vendor/KDTree/`：原始 KDTree Baseline 压缩包与只读镜像。
- `vendor/Alacarte.zip` / `vendor/Alacarte/`：原始 Alacarte 生成器压缩包与只读镜像。
- `include/`：为 `isrjs_kds.hpp` 补齐的最小兼容接口层。
- `src/`：实验驱动、数据加载、指标采集、结果导出。
- `python/`：数据生成、结果聚合、图表与 Markdown 报告脚本。
- `scripts/`：完整实验计划调度与 vendor 完整性校验。
- `docs/original/`：原始原理文档与实验计划文档。
- `results/`：运行产物目录。

## 一键运行

默认执行**完整实验网络**：

```bash
./run
```

快速 smoke 验证：

```bash
./run --smoke
```

也可以用：

```bash
./run.sh --smoke
```

可选环境变量：

```bash
JOBS=8 ./run
```

`run` 会自动完成以下步骤：

1. 校验 vendor 文件哈希；
2. 编译 `kdtree_benchmark` 与 `isrjs_smoke`；
3. 执行 `isrjs_kds.hpp` 原样 smoke 运行；
4. 按实验网络生成数据集并运行基准；
5. 聚合结果、导出 CSV、绘制图表、生成 `report.md`。

## 实验网络

### 1. 正确性校验网络

- `d ∈ {2,3,4,5}`
- `N = 2000`
- `alpha_target ∈ {0.1, 10, 100}`
- `t = 2e5`
- `F ∈ {F0, F1}`
- `rep = 0`
- 额外执行精确 join 物化，检查：
  - `W = Σ_r w(r)`
  - `M_point(r) + M_subtree(r) = w(r)`
  - 样本合法性
  - 空结果一致性
  - 经验频率偏差指标

### 2. 主行为网络

#### 密度 sweep

- `N = 50000`
- `t = 2e5`
- `alpha_target ∈ {0.1, 1, 10, 100}`
- `d ∈ {2,3,4,5}`
- `F ∈ {F0,F1}`
- `rep ∈ {0,1,2}`

#### 输入规模 sweep

- `alpha_target = 10`
- `t = 2e5`
- `N ∈ {10000, 50000, 100000}`
- `d ∈ {2,3,4,5}`
- `F ∈ {F0,F1}`
- `rep ∈ {0,1,2}`

#### 采样规模 sweep

- `N = 50000`
- `alpha_target = 10`
- `t ∈ {1e4, 5e4, 2e5, 1e6}`
- `d ∈ {2,3,4,5}`
- `F ∈ {F0,F1}`
- `rep ∈ {0,1,2}`

### 3. 分块补充网络

- `d ∈ {2,5}`
- `N = 20000`
- `alpha_target = 10`
- `t ∈ {5e5, 2e6, 3e6}`
- `F ∈ {F0,F1}`
- `rep ∈ {0,1}`

## 数据族映射

本仓库把你文档中的族映射为：

- `F0`：`volume_dist="fixed"`, `shape_sigma=0.0`
- `F1`：`volume_dist="fixed"`, `shape_sigma=0.75`

并且统一使用 preset `TH`：

- `dtype=float32`
- `tune_samples=200000`
- `tune_tol_rel=0.02`
- `tune_max_iter=30`
- `chunk_size=2000000`

后续提到的 `alpha` 全部按 `alpha_target` 解释。

## 已采集指标

### 构建阶段

- `T_lift`, `T_tree`, `T_build`
- `H_tree`, `max_depth`, `avg_node_depth`
- 按深度分层的 bbox 边长 / 体积分位数（`bbox_by_depth.csv`）

### 计数阶段

- `T_count_query`, `T_left_dist`, `T_count`
- `W`, `alpha_actual`
- `w(r)`
- `V_count(r)`, `P_count(r)`, `C_count(r)`, `H_count(r)`
- `S_prune(r)`, `S_contain(r)`, `D_count_max(r)`
- 均值 / 中位数 / p90 / p99 汇总

### 查询分解阶段

- `V_search(r)`, `P_search(r)`
- `B_mid(r)`, `B_leaf(r)`, `B_subtree(r)`
- `m(r)`, `M_point(r)`, `M_subtree(r)`
- `rho(r)`, `sigma(r)`, `avg_subtree_mass(r)`
- 左盒等权视角与按 `w(r)/W` 加权视角两套汇总

### 采样阶段

- `T_plan_left`, `T_group`, `T_decompose`, `T_local_dist`, `T_emit`, `T_sample`
- `samples_per_sec`, `ns_per_sample`
- `U_t`, `G_t`, `avg_repeat_global`, `avg_group_size_exec`
- `group_size` 分布与直方图
- `K_point(g)`, `K_subtree(g)`
- 组等权与按组大小加权两套汇总

## 输出产物

运行结束后，结果落在：

- `results/full/` 或 `results/smoke/`

其中：

- `plan_manifest.csv`：计划清单
- `datasets/`：二进制数据集与元数据
- `runs/<run_id>/metrics.json`：单次 run 详细指标
- `runs/<run_id>/bbox_by_depth.csv`
- `runs/<run_id>/group_size_hist.csv`
- `runs/<run_id>/left_metrics_sample.csv`
- `runs/<run_id>/group_metrics_sample.csv`
- `analysis/runs_flat.csv`：扁平总表
- `analysis/bbox_depth_all.csv`
- `analysis/group_size_hist_all.csv`
- `analysis/correctness_summary.csv`
- `analysis/*.png`
- `analysis/report.md`

## 额外说明

- `build/isrjs_smoke` 用来确认 `vendor/KDTree/isrjs_kds/isrjs_kds.hpp` 在本仓库中**原样可编译可运行**。
- 主实验驱动直接使用 `vendor/KDTree/utils/kdtree.hpp` 与 `vendor/KDTree/utils/weighted_sampling.hpp`，并在外层做同语义指标采集，因此不会改动基线内部逻辑。
- `scripts/verify_vendor_integrity.py` 会校验 vendor 区文件哈希，确保 baseline 与 generator 均保持只读状态。
