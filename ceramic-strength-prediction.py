"""
陶瓷工艺参数→强度预测：三模型对比 + 5折交叉验证
特征：temperature, holding_time, additive_ratio, density, porosity
目标：strength
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "material_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["sintering_temp", "holding_time", "additive_ratio"]
TARGET = "strength"

# 1. 读取数据 + 清洗
df = pd.read_csv(DATA_PATH)
print("=" * 55)
print(f"数据加载完成：{len(df)} 条样品，{len(df.columns)} 个字段")
print(f"输入特征：{FEATURES}")
print(f"目标变量：{TARGET}")

# 清洗：删除重复行，数值列缺失值用列均值填充
df = df.drop_duplicates()
numeric_cols = df[FEATURES + [TARGET]].select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print(f"清洗后：{len(df)} 条样品")
print("=" * 55)

X = df[FEATURES]
y = df[TARGET]

# 2. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集：{len(X_train)} 条 | 测试集：{len(X_test)} 条\n")

# 3. 三个模型
models = {
    "线性回归": LinearRegression(),
    "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
}

# 4. 5折交叉验证
print("=" * 55)
print("5折交叉验证结果（MAE，越低越好）")
print("=" * 55)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv,
                             scoring="neg_mean_absolute_error")
    mae_mean = -scores.mean()
    mae_std = scores.std()
    cv_results[name] = (mae_mean, mae_std)
    print(f"  {name:8s} | MAE = {mae_mean:.2f} ± {mae_std:.2f}")

# 5. 选最佳模型，在测试集上验证
best_name = min(cv_results, key=lambda x: cv_results[x][0])
best_model = models[best_name]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"\n{'=' * 55}")
print(f"最佳模型：{best_name}")
print(f"测试集 MAE = {test_mae:.2f} MPa")
print(f"测试集 R²  = {test_r2:.4f}")
print("=" * 55)

# 6. 特征重要性
print(f"\n特征重要性（{best_name}）：")
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
        print(f"  {feat:18s}: {imp:.4f}")

    plt.figure(figsize=(8, 5))
    sorted_pairs = sorted(zip(FEATURES, importances), key=lambda x: x[1])
    names_sorted = [p[0] for p in sorted_pairs]
    vals_sorted = [p[1] for p in sorted_pairs]
    plt.barh(names_sorted, vals_sorted, color="steelblue")
    plt.xlabel("Feature Importance")
    plt.title(f"Feature Importance - {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    print(f"\n图已保存：outputs/feature_importance.png")
else:
    print("  线性回归系数：")
    for feat, coef in zip(FEATURES, best_model.coef_):
        print(f"  {feat:18s}: {coef:.4f}")

# 7. 预测值 vs 真实值
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, s=60, color="steelblue", edgecolors="white")
lims = [min(y.min(), y_pred.min()) - 5, max(y.max(), y_pred.max()) + 5]
plt.plot(lims, lims, "r--", lw=2, label="理想预测线")
plt.xlabel("真实强度 (MPa)")
plt.ylabel("预测强度 (MPa)")
plt.title(f"{best_name}：预测 vs 真实\nMAE={test_mae:.2f} MPa, R²={test_r2:.4f}")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "prediction_vs_actual.png", dpi=150)
print(f"图已保存：outputs/prediction_vs_actual.png")

# 8. 三模型测试集对比
all_test_mae = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    all_test_mae[name] = mean_absolute_error(y_test, pred)

plt.figure(figsize=(7, 4))
names = list(all_test_mae.keys())
maes = list(all_test_mae.values())
bars = plt.bar(names, maes, color=["#95a5a6", "#3498db", "#e74c3c"])
plt.ylabel("测试集 MAE (MPa)")
plt.title("三模型测试集误差对比（越低越好）")
for bar, mae in zip(bars, maes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{mae:.2f}", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
print(f"图已保存：outputs/model_comparison.png")

print(f"\n{'=' * 55}")
print("全部完成！去 outputs/ 文件夹看结果")
print("=" * 55)
