# 2_generate_figures.py

# --- 导入所需库 ---
import pandas as pd  # 主要用于创建DataFrame（如特征重要性）
import joblib  # 用于加载脚本1保存的.pkl文件
import os  # 用于处理文件路径
import matplotlib.pyplot as plt  # Python主要的绘图库
import seaborn as sns  # 基于Matplotlib的高级绘图库，提供更美观的图表
import shap  # 用于绘制SHAP分析图
from sklearn.metrics import ConfusionMatrixDisplay  # Scikit-learn中用于绘制混淆矩阵的便捷工具

# --- 全局配置与路径设置 ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    project_root = os.getcwd()
OUTPUT_DIR = os.path.join(project_root, 'output')
FIGURES_DIR = os.path.join(project_root, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)  # 创建图片文件夹

# --- 步骤 1: 加载所有预计算的“工件” ---
# 从output文件夹加载脚本1中保存的所有结果
stats = joblib.load(os.path.join(OUTPUT_DIR, 'statistical_results.pkl'))
model = joblib.load(os.path.join(OUTPUT_DIR, 'best_model.pkl'))
le = joblib.load(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))
test_data = joblib.load(os.path.join(OUTPUT_DIR, 'test_data.pkl'))
shap_values = joblib.load(os.path.join(OUTPUT_DIR, 'shap_values.pkl'))

# 从加载的数据中解包出测试集
X_test, y_test = test_data['X_test'], test_data['y_test']
print("All artifacts loaded. Generating figures...")

# --- 图 1: 模型性能对比图 ---
plt.style.use('seaborn-v0_8-whitegrid')  # 使用一种美观的网格样式
fig, ax = plt.subplots(figsize=(10, 8))  # 创建一个图形和坐标轴
bar_labels = ['Elemental Features', 'Enhanced Features']
means = [stats['mean_elemental'], stats['mean_enhanced']]  # 提取均值
stds = [stats['std_elemental'], stats['std_enhanced']]      # 提取标准差作为误差棒
bars = ax.bar(bar_labels, means, yerr=stds, capsize=5, color=['steelblue', 'indianred'], edgecolor='black', alpha=0.8)
ax.set_ylim(0.75, 0.90)  # 设置Y轴范围以突出差异
ax.set_ylabel('Mean Accuracy', fontsize=14)
ax.set_title('Model Performance Comparison (10-fold, 3-repeat Cross-Validation)', fontsize=16)
# 在每个条形图上方标注精确的均值
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom', fontsize=12)
# 绘制表示统计显著性检验的横线和p值
ax.plot([0, 1], [0.89, 0.89], lw=1.5, color='black')
ax.text(0.5, 0.892, f"p = {stats['p_value']:.4f}", ha='center', va='bottom', fontsize=12)
plt.savefig(os.path.join(FIGURES_DIR, '1_performance_comparison.png'), dpi=300)
print("Figure 1 saved.")
plt.close()  # 关闭当前图形，防止在后续绘图中重叠

# --- 图 2: 特征重要性图 ---
plt.style.use('default')  # 恢复默认样式
# 从训练好的模型中提取特征重要性，并创建DataFrame进行排序
importance_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(14, 10))
# 使用seaborn绘制水平条形图，并使用'rocket_r'调色板
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='rocket_r')
plt.title('Feature Importance (Best Model on Elemental Set)', fontsize=16)
plt.tight_layout()  # 自动调整布局以防标签重叠
plt.savefig(os.path.join(FIGURES_DIR, '2_feature_importance.png'), dpi=300)
print("Figure 2 saved.")
plt.close()

# --- 图 3: 混淆矩阵 ---
fig, ax = plt.subplots(figsize=(10, 10))
# 使用Scikit-learn的便捷函数，直接从模型和测试数据生成混淆矩阵图
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=le.classes_, cmap='cividis', normalize=None, ax=ax)
ax.set_title('Confusion Matrix (Best Model on Elemental Set)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '3_confusion_matrix.png'), dpi=300)
print("Figure 3 saved.")
plt.close()

# --- 图 4: SHAP 总结图 (Bar Plot) ---
plt.figure()
# 调用SHAP库的绘图函数，指定plot_type="bar"来创建全局特征重要性的条形图
shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=le.classes_, show=False)
plt.title('SHAP Summary Plot', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '4_shap_summary_bar.png'), dpi=300, bbox_inches='tight')
print("Figure 4 saved.")
plt.close()

# --- 图 5: SHAP 总结图 (Beeswarm Plot for 'Im' Phase) ---
# 从标签编码器中找到 'Im' 相对应的数字索引
im_class_index = list(le.classes_).index('Im')
plt.figure()
# 再次调用summary_plot，但这次只传入'Im'相的SHAP值。
# 默认的plot_type会生成蜂群图（beeswarm plot）。
shap.summary_plot(shap_values[im_class_index], X_test, show=False)
plt.title('SHAP Summary Plot for Intermetallic (Im) Phase', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '5_shap_beeswarm_Im.png'), dpi=300, bbox_inches='tight')
print("Figure 5 saved.")
plt.close()

