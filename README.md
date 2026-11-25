# 局部遮挡下的点集双向距离和配准放行

`data/record.json` 是 局部遮挡下的点集双向距离和配准放行 使用的本地合成交换样本。数组按 JSON 的下标顺序排列。本文的坐标、单位和枚举约定与该记录配套。记录中如果有 `scale` 或 `limits`，对应同一输入协议的部署规格。

points 每行是稳定 ID、区域、三维点、单位法向及置信权重。bbox 是以块本地坐标给出的闭包围盒。views 给出每台相机在自己坐标里的闭 AABB，候选只在同区域内配对。region_rules 每行为最小法向内积、最低覆盖率、平方距离上限和分位比例。pose_right_to_left 用标准右手旋转将右相机坐标变到左相机坐标。arrival 是按块名排的某次送入次序，point_sets 中的块名组成该次快照的全部预期块。

## 读取样本

使用 Python 3.11 或 3.12，并在虚拟环境安装锁定依赖。读取入口返回原始记录。作为命令运行时，它会列出各顶层输入项目以及 NumPy 版本。

```sh
python -m pip install -r requirements.txt
python -m hausdorff_kernel_input
python -m unittest discover -s tests -v
```

测试读取交换样本，核对其中的张量尺寸、标识与数据约定。

## 距离算子

实现位于 `hausdorff_kernel.py`，只依赖 NumPy。标定脚本和训练代码共用同一套
区域、视域、平局、统计和梯度语义。

```python
import hausdorff_kernel as hk

# 完整样本；pose 可选 "p0"/"p1" 或下标，through 可只看到达前缀。
report = hk.evaluate_record(pose_id="p0")
report.approved                  # 所有配置区域双向达标且最终时才为 True
report.released_regions()        # ["seam"]；key 为 missing_evidence

partial = hk.evaluate_record(pose_id="p0", through=2)
partial.regions["seam"].temporary_no_neighbor
```

通用扫描可构造 `PointChunk` + `PointFrame`，或使用可复用的
`PointDistanceOperator`。每个位姿都通过新的 `DistanceSession` 重算视域、
候选合法性、近邻和放行资格，不会复用上一位姿的结论：

```python
op = hk.PointDistanceOperator.from_record(record)
r0 = op.evaluate_pose("p0")
r1 = op.evaluate_pose("p1")       # 全新会话，不沿用 p0 的资格
```

### 判定口径

- 距离为平方欧氏距离；报告同时给出 `distance = sqrt(distance_sq)`。
- 候选必须同区域、法向内积不低于规则阈值，并且通过两侧闭视域的共同可见判定。
- 正反向最近邻独立计算；键为 `(distance_sq, stable_id)`，ID 按稳定字符串排序。
- 相同稳定 ID 的重叠块副本只计一次权重；坐标、区域、法向或权重不一致会把对应
  区域置为 `rejected`，不会放行。
- 加权分位是累计权重首次达到 `quantile * 匹配权重` 的最小距离。
- 覆盖率分母是共同可见且有效的查询权重，分子是找到合法匹配的查询权重。
- 无合法邻居的最终区域状态为 `missing_evidence`，分位为 `None`，不是零。
- 全局/区域块未到齐时为 `partial` + `temporary_no_neighbor`，不能提前判遮挡。
- 双向 `coverage >= min_coverage` 且 `quantile_sq <= max_distance_sq` 时，该区域
  才 `released`；所有 `region_rules` 区域放行后位姿才 `approved`。

### 内存与包围盒排除

算子只保存每个唯一查询点的当前最佳候选和增量统计，不保存 `N_left × N_right`
距离矩阵。候选按 `MemoryBudget.scratch_bytes` 分块计算临时距离向量，
`point_chunk_max` 对应 `MemoryBudget.max_points`。

候选块包围盒先把查询点逆变换到候选块本地坐标，再计算点到闭 AABB 的距离下界。
只有下界严格大于现任距离（容差内）才跳过；下界等于现任距离时仍可能出现同距离、
更小稳定 ID 的候选，因此不会提前跳过。每条跳过生成 `Exclusion`，保存位姿、查询点、
包围盒、下界和现任候选；`exclusion.recheck(new_pose)` 会按新位姿重算依据。

大于内存的输入可使用 `ChunkHandle(chunk_id, side, bbox, loader=...)` 按需读块；
最终匹配和排除依据可用 `report.write_matches_jsonl(path)` 与
`report.write_exclusions_jsonl(path)` 逐行写出。

### 训练导数

只用已放行区域的冻结近邻图：

```python
grad = hk.distance_loss_and_gradients(
    report,
    reduction="weighted_mean",   # 或 "sum" / "mean"
    weight_mode="query",          # 或 "query_neighbor" / "none"
)
grad.point_gradients["left"]["l1"]
grad.point_gradients["right"]["r04"]
grad.pose_gradient["translation"]
grad.pose_gradient["angle_z_deg"]
grad.sources                      # 每条边的来源和局部贡献
grad.stable                       # 等距/法向边界/视域边界会置 False
```

同一个点作为多个查询的邻居时，各边贡献按稳定 ID 汇总。精确等距候选、法向阈值
边界或闭视域边界会在对应 `Match` / `GradientSource` 上标出
`equidistant`、`normal_boundary`、`visibility_boundary`。默认返回所选分支导数并
把 `stable/trainable` 置为 `False`；`require_stable=True` 或
`training_update_plan(..., allow_unstable=False)` 会抛出
`UnstableSupportError`，避免静默训练。

运行自带行为测试：

```sh
python -m unittest discover -s tests -v
```
