# 2025年西北太平洋台风路径预报评估

Reproducible evaluation of GraphCast, Pangu-Weather and FourCastNet tropical-cyclone track forecasts.

本项目整理了论文《三种人工智能气象模型的台风路径预报比较与误差诊断》的分析程序、归档中心坐标、统计结果、图件和论文文稿。研究在相同ERA5初始化、相同起报时刻、相同JMA验证资料和统一中心追踪规则下，对GraphCast、Pangu-Weather与FourCastNet的2025年西北太平洋台风路径预报进行比较。

![三模型平均路径误差](figures/Figure_1_primary.png)

## 研究问题与主要结果

项目依次回答三个问题：

1. 三个模型的路径误差如何随预报时效变化？
2. 台风路径转向前后，误差大小和方向如何改变？
3. 三模型预测位置的分歧能否提示较大的共识路径误差？

主分析包含14个台风、74个完整120 h起报案例。每个案例在6–120 h内每6 h验证一次，三个模型共形成4,440条“模型×案例×时效”记录。另有13个台风各保留一个36–108 h截短案例，用于共同短时效补充分析。

主分析显示：GraphCast在6–66 h的样本平均DPE最小，Pangu-Weather在72–120 h最小；GraphCast相对FourCastNet的优势在54–114 h获得同时置信带支持，Pangu-Weather相对FourCastNet的优势在54–120 h获得支持。误差方向分析显示，长时效误差主要表现为沿台风运动方向的前后错位。转向前后配对分析中，三个模型转向后的样本平均DPE均高于转向前，平均增量分别为93.3、58.8和117.8 km。模型分歧与共识误差的关联随预报时效变化，24 h和48 h的正相关最清楚。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `inputs/` | JMA最佳路径、案例目录和三模型归档中心坐标，是第四章统计复现的固定输入 |
| `data/` | 第一、三、五章及附录使用的处理后数据，包括`overall_by_lead.csv`、`paired_by_lead.csv`、`IMD_correlations.csv`等 |
| `tables/` | 第四章生成的记录级表、阶段配对、分组差、交互和敏感性结果 |
| `code/` | RMSE、路径方向、转向、分组和模型分歧分析程序 |
| `code/upstream/` | 原中心追踪、总体误差和模型配对分析程序的归档副本 |
| `supplement_13/` | 另外13个截短台风的共同6–36 h补充分析 |
| `figures/` | 论文主要图件，提供PNG预览及部分PDF、SVG矢量文件 |
| `audit/` | 软件版本、随机种子、输入哈希和独立核对结果 |
| `manuscript/` | 当前Markdown论文全文 |
| `docs/` | 分析方案、数据字典和复现边界说明 |

## 快速复现

建议使用Python 3.12：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

`scripts/reproduce.py`从仓库中的预报中心坐标开始，依次重算距离RMSE、第四章路径变化分析、13个截短台风补充分析和第五章风险提示图。各分析也可单独运行，例如：

```bash
python code/derive_rmse.py
python code/analyze_chapter4.py
python code/verify_chapter4.py
python code/plot_chapter4.py
python code/plot_chapter5_focused.py
```

数据字段、计算层级和主要输出见[数据字典](docs/DATA_DICTIONARY.md)，能够复现的范围及上游文件要求见[复现说明](docs/REPRODUCIBILITY.md)。

## 计算口径

主分析先在每个台风内部平均其多个起报案例，再对14个台风等权平均。模型比较固定同一台风、同一起报时刻和同一预报时效，计算两模型DPE差值。置信区间以台风为重采样单位，保留同一台风在模型和时效之间的相关结构。转向前后分析同样先在台风内部匹配相同预报时效，再对台风等权汇总。

## 数据来源

- 验证资料：Japan Meteorological Agency, RSMC Tokyo Best Track Data。
- 模型：GraphCast、Pangu-Weather、FourCastNet。
- 初始化资料：ERA5再分析资料。

JMA来源文件快照及SHA-256记录在`inputs/`中。模型论文、统计方法和业务验证文献列于[论文文稿](manuscript/AI台风路径论文.md)的参考文献部分。

## 复现范围

仓库可以从已归档的三模型预测中心位置重新计算统计表和图件。模型权重、ERA5输入和区域GRIB预报场体积较大，未纳入Git仓库；从原始气象场重新运行模型和中心追踪时，需要另行准备这些文件。`code/upstream/`保留了对应的追踪与上游分析程序，便于核对算法和参数。

