# 实现口径说明

这份说明只解释实现时必须明确但原计划里未完全形式化的口径。

## 1. `KDTree.zip` 的使用方式

- `vendor/KDTree.zip` 与 `vendor/KDTree/` 都按原样保留。
- 实验驱动直接复用：
  - `vendor/KDTree/utils/kdtree.hpp`
  - `vendor/KDTree/utils/weighted_sampling.hpp`
- `vendor/KDTree/isrjs_kds/isrjs_kds.hpp` 不做任何改动，并通过 `build/isrjs_smoke` 原样编译验证。

## 2. 查询与采样语义

实验驱动严格使用与原文一致的语义：

- 右侧盒做 `2d` lifting；
- 左侧查询使用 `nextDown(hi)` / `nextUp(lo)` 转成闭区间；
- 计数递归使用 `disjoint / contained / partial` 三分支；
- 查询分解使用：
  - 内部节点中位点单点 item
  - 被整体覆盖的孩子子树 item
  - 叶子单点 item
- 采样按：
  - 先按 `w(r)` 抽左端
  - 再在固定左端条件下按 item 权重抽样
  - 完整子树 item 内部再均匀抽取一个点

## 3. 指标口径补充

### 3.1 `tree_height`

- `max_depth`：根深度为 `0`
- `tree_height = max_depth + 1`

### 3.2 `L_bbox(q)`

计划中 `L_bbox(q)` 只给出“边长分位数”，未固定是“逐节点”还是“逐维”。
本实现采用：

- 对每个深度层，把该层所有节点在所有维度上的 bbox 边长全部展开；
- 在这批长度上计算 `p50/p90/p99`。

因此 `bbox_by_depth.csv` 中的 `edge_p50/p90/p99` 反映的是“节点-维度”展开后的边长分布。

### 3.3 `V_bbox(q)`

- 对每个节点计算其 lifted bbox 体积；
- 再按深度层汇总 `p50/p90/p99`。

### 3.4 加权统计

- 左盒加权视角使用权重 `w(r)`，等价于 `p_r = w(r)/W` 的未归一化形式；
- 组级加权视角使用权重 `|g|`；
- 对 `rho`, `sigma`, `avg_subtree_mass` 等只在部分点定义的指标，未定义值以 `NaN` 处理，并在汇总时自动忽略。

### 3.5 时间边界

- `T_build / T_count / T_sample` 都从数据读入完成之后开始；
- 文件读写、CSV 导出、图表绘制都不计入阶段时间；
- 样本合法性复核与 correctness 物化 join 也不计入 `T_sample`。

## 4. correctness 额外检查

在 `correctness` 网络中额外执行：

- 精确物化 join；
- 样本经验 pair 频率与理论均匀分布之间的：
  - `pair_tv_to_uniform`
  - `pair_linf_to_uniform`
- 左端边缘分布与理论 `w(r)/W` 的：
  - `left_tv_to_theory`

这些指标用于 sanity check，不作为严格统计检验结论。
