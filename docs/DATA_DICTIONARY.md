# 数据字典

## 主要输入

| 文件 | 粒度 | 作用 |
|---|---|---|
| `inputs/case_catalog.csv` | 一个台风×一个起报时刻 | 记录案例编号、台风编号、起报时间、预报长度和主/补充样本类型 |
| `inputs/track_points.csv` | 一个模型×一个案例×一个预报时效 | 保存预测中心、JMA中心和直接路径误差，是第四章的记录级输入 |
| `inputs/jma_track_points.csv` | 一个台风×一个JMA时次 | 保存JMA最佳路径位置、等级和时间，用于位置验证及转向识别 |
| `data/error_components_cases.csv` | 一个模型×一个案例×一个预报时效 | 保存DPE、ATE、CTE及案例属性，用于总体误差和方向分析 |
| `data/disagreement_cases.csv` | 一个案例×一个预报时效 | 保存三模型分歧IMD和共识路径误差 |

## 关键输出

| 文件 | 内容 |
|---|---|
| `data/overall_by_lead.csv` | 6–120 h逐时效、台风等权的平均DPE及逐项区间 |
| `data/rmse_by_lead.csv` | 逐时效距离RMSE及台风重采样区间 |
| `data/paired_by_lead.csv` | 三组模型配对DPE差值、逐项区间和60项同时置信带 |
| `tables/overall_by_lead.csv` | DPE、绝对/有符号ATE和CTE的逐时效统计 |
| `tables/event_matched_lead_differences.csv` | 同一台风、同一预报时效的转向后减转向前差值 |
| `tables/event_primary_inference.csv` | 三模型×三指标的转向前后平均差及置信区间 |
| `tables/group_difference.csv` | 转向组减未转向组的逐时效差值及区间 |
| `tables/group_interactions.csv` | 模型配对差在两类路径之间的交互 |
| `data/IMD_correlations.csv` | 各时效IMD与共识DPE的相关系数及区间 |
| `data/IMD_LOCO_metrics.csv` | 留出台风风险识别的AUC、POD、FAR、精确率和提升倍数 |

## 主要字段

| 字段 | 含义 |
|---|---|
| `TC_ID` | JMA台风编号 |
| `case_id` | 一个台风在一个起报时刻对应的案例编号 |
| `model` | `graphcast`、`pangu`或`fourcastnet` |
| `lead_hours` | 从起报时刻开始计算的预报时效，单位h |
| `track_error_km` / `DPE` | 预测中心与JMA中心的大圆距离，单位km |
| `ATE` | 沿轨误差；正值表示超前，负值表示滞后 |
| `CTE` | 横轨误差；正值表示运动方向右侧，负值表示左侧 |
| `absATE` / `absCTE` | 沿轨和横轨误差的绝对值 |
| `IMD` | 三个预测中心两两距离的平均值，表示模型分歧 |
| `consensus_DPE` | 三模型球面平均中心到JMA中心的距离 |

## 样本层级

- 主分析：14个台风、74个完整120 h案例、20个评分时效。
- 补充分析：另外13个台风各一个36–108 h截短案例。
- 全部可用样本：27个台风、87个案例，对应71个唯一起报任务。

