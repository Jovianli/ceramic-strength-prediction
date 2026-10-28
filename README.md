# 陶瓷工艺参数→抗压强度预测模型

基于 Python + scikit-learn 的陶瓷工艺参数回归建模项目，对比线性回归、随机森林、XGBoost 三种模型，为陶瓷工艺优化提供数据参考。

## 项目背景

陶瓷制备过程中，烧结温度、保温时间、添加剂比例等工艺参数直接影响成品强度。传统工艺优化依赖试错，成本高、周期长。本项目通过建立工艺参数→强度的回归模型，帮助工艺工程师缩小试错范围。

## 数据说明

- 样本量：60 组陶瓷工艺实验数据
- 输入特征（3个）：
  - `sintering_temp`：烧结温度（℃），范围 1200~1600
  - `holding_time`：保温时间（h），范围 1~4
  - `additive_ratio`：添加剂比例（%），范围 0~5
- 目标变量：`strength`（抗压强度，MPa）

## 方法

1. 数据清洗：删除重复行，数值列缺失值用均值填充
2. 训练集/测试集划分：80% / 20%（48条训练，12条测试）
3. 5折交叉验证评估模型稳定性
4. 三模型对比：
   - 线性回归（Linear Regression）
   - 随机森林（Random Forest, n_estimators=100）
   - XGBoost（n_estimators=100）
5. 评价指标：MAE（平均绝对误差）、R²

## 实验结果

| 模型 | 5折交叉验证 MAE | 测试集 MAE | R² |
|---|---|---|---|
| **线性回归** | **19.79 ± 2.11** | **17.49 MPa** | **0.8201** |
| 随机森林 | 30.57 ± 4.18 | — | — |
| XGBoost | 28.84 ± 8.97 | — | — |

### 关键发现

- **线性回归表现最优**：MAE = 17.49 MPa，R² = 0.82
- 树模型（随机森林/XGBoost）在小样本（60条）上过拟合，表现不如线性回归
- **特征重要性排序**（线性回归系数）：
  1. 保温时间（系数 43.8）——对强度影响最大
  2. 添加剂比例（系数 15.0）——次之
  3. 烧结温度（系数 0.30）——影响最小（在1200-1600℃范围内）

## 输出图表

- `outputs/model_comparison.png`：三模型 MAE 对比柱状图
- `outputs/prediction_vs_actual.png`：预测值 vs 真实值散点图

## 如何运行

```bash
# 安装依赖
pip install pandas scikit-learn xgboost matplotlib

# 运行
python ceramic-strength-prediction.py
```

## 技术栈

Python · pandas · scikit-learn · XGBoost · matplotlib
