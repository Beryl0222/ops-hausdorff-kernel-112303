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
