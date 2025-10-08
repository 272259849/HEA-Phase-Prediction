# 生成各种直观的图
#图 1: 模型性能对比
#图 2: 元素特征重要性
#图 3: 混淆矩阵
#图 4: SHAP 总结图（条形图）
#图 5: SHAP 总结图（蜂窝）

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import ConfusionMatrixDisplay


try:
    project_root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    project_root = os.getcwd()
OUTPUT_DIR = os.path.join(project_root, 'output')
FIGURES_DIR = os.path.join(project_root, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Load All Artifacts
stats = joblib.load(os.path.join(OUTPUT_DIR, 'statistical_results.pkl'))
model = joblib.load(os.path.join(OUTPUT_DIR, 'best_model.pkl'))
le = joblib.load(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))
test_data = joblib.load(os.path.join(OUTPUT_DIR, 'test_data.pkl'))
shap_values = joblib.load(os.path.join(OUTPUT_DIR, 'shap_values.pkl'))

X_test, y_test = test_data['X_test'], test_data['y_test']
print("All artifacts loaded. Generating figures...")

# Fig1: Model Performance Comparison
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 8))
bar_labels = ['Elemental Features', 'Enhanced Features']
means = [stats['mean_elemental'], stats['mean_enhanced']]
stds = [stats['std_elemental'], stats['std_enhanced']]
bars = ax.bar(bar_labels, means, yerr=stds, capsize=5, color=['steelblue', 'indianred'], edgecolor='black', alpha=0.8)
ax.set_ylim(0.75, 0.90)
ax.set_ylabel('Mean Accuracy', fontsize=14)
ax.set_title('Model Performance Comparison (10-fold, 3-repeat Cross-Validation)', fontsize=16)
for i, bar in enumerate(bars):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom', fontsize=12)
ax.plot([0, 1], [0.89, 0.89], lw=1.5, color='black')
ax.text(0.5, 0.892, f"p = {stats['p_value']:.4f}", ha='center', va='bottom', fontsize=12)
plt.savefig(os.path.join(FIGURES_DIR, '1_performance_comparison.png'), dpi=300)
print("Figure 1 saved.")
plt.close()

#Fig2: Feature Importance
plt.style.use('default')
importance_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(14, 10))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='rocket_r')
plt.title('Feature Importance (Best Model on Elemental Set)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '2_feature_importance.png'), dpi=300)
print("Figure 2 saved.")
plt.close()

#Fig3: Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 10))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=le.classes_, cmap='cividis', normalize=None, ax=ax)
ax.set_title('Confusion Matrix (Best Model on Elemental Set)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '3_confusion_matrix.png'), dpi=300)
print("Figure 3 saved.")
plt.close()

#Fig4: SHAP Summary (Bar Plot)
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=le.classes_, show=False)
plt.title('SHAP Summary Plot', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '4_shap_summary_bar.png'), dpi=300, bbox_inches='tight')
print("Figure 4 saved.")
plt.close()

#Fig5: SHAP Summary for 'Im' Phase (Beeswarm Plot)
im_class_index = list(le.classes_).index('Im')
plt.figure()
shap.summary_plot(shap_values[im_class_index], X_test, show=False)
plt.title('SHAP Summary Plot for Intermetallic (Im) Phase', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '5_shap_beeswarm_Im.png'), dpi=300, bbox_inches='tight')
print("Figure 5 saved.")
plt.close()
