# 复现说明

## 可直接复现的内容

仓库保存了三模型的预测中心位置、JMA验证位置、案例目录和分析程序，因此可以重新计算：

- 逐时效平均DPE和距离RMSE；
- 三组模型的配对误差差值及置信区间；
- ATE/CTE误差分解；
- 转向识别、转向前后配对和转向/未转向分组比较；
- 三模型分歧与共识路径误差的相关和留出台风风险识别；
- 正文与补充分析图件。

`python scripts/reproduce.py`会运行上述统计复现流程。每个阶段的输入、随机种子、重复次数和输出核对记录保存在`audit/`、`data/*audit.json`及`supplement_13/audit/`。

## 上游文件

GraphCast、Pangu-Weather和FourCastNet的模型权重、ERA5初始化资料以及区域GRIB预报场未纳入仓库。它们用于生成`inputs/track_points.csv`和`data/error_components_cases.csv`中的预测中心。需要从气象场重新追踪中心时，可使用`code/upstream/`中的归档程序，并按原模型要求准备相应资料。

## 完整性检查

仓库根目录的`MANIFEST.sha256`记录所有版本化文件的SHA-256。运行以下命令可核对文件是否被修改：

```bash
shasum -a 256 -c MANIFEST.sha256
```

`code/verify_chapter4.py`和`supplement_13/code/verify_short.py`会独立检查样本数量、配对关系、距离计算、输出表结构和主要统计结果。

