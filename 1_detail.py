# 1_train_and_evaluate.py

# --- 导入所需库 ---
import pandas as pd  # 用于数据处理，核心是DataFrame
import numpy as np  # 用于数值计算，特别是统计（均值、标准差）
import joblib  # 用于高效地保存和加载Python对象（如模型、数据）
import os  # 用于处理文件路径，确保跨平台兼容性
import shap  # 用于计算SHAP值，解释模型预测
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score  # Scikit-learn模型选择工具
from sklearn.preprocessing import LabelEncoder  # 用于将文本标签（如'FCC'）转换为数字
from sklearn.impute import SimpleImputer  # 用于处理数据中的缺失值（NaN）
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器模型
from scipy.stats import ttest_rel  # 导入用于统计检验的配对t检验函数

# --- 全局配置 ---
# 设置随机种子为42，确保所有随机操作（如数据分割、模型训练）的结果都是可复现的。
# 这是科学研究中至关重要的一步。
RANDOM_STATE = 42

# --- 路径设置 ---
# 定义项目根目录和输出目录，确保脚本在任何位置运行都能正确找到文件。
# try-except块使其在标准Python脚本和Jupyter Notebook等环境中都能稳健工作。
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    project_root = os.getcwd()
OUTPUT_DIR = os.path.join(project_root, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 创建输出文件夹，如果已存在则忽略

# --- 步骤 1: 数据加载与预处理 ---
# 从项目根目录加载Excel数据文件到pandas DataFrame中
df = pd.read_excel(os.path.join(project_root, 'HEA_DataSet.xlsx'))

# 定义论文中提到的两组特征列和一个目标列
elemental_cols = ['Al', 'Co', 'Cr', 'Fe', 'Ni', 'Cu', 'Mn', 'Ti', 'V', 'Nb', 'Mo', 'Zr', 'Hf', 'Ta', 'W', 'C', 'Mg', 'Zn', 'Si', 'Re', 'N', 'Sc', 'Li', 'Sn', 'Be']
physical_cols = ['VEC', 'dSmix', 'dHmix', 'Atom.Size.Diff', 'Elect.Diff', 'Density_calc', 'Tm']
enhanced_cols = elemental_cols + physical_cols  # 增强特征集 = 元素特征 + 物理特征
target_col = 'Phases'

# 将目标列（如 'BCC_SS', 'FCC_SS'）从文本转换为数字（如 0, 1）
le = LabelEncoder()
y_encoded = le.fit_transform(df[target_col])

# 准备两个特征矩阵(X)供后续比较
X_elemental = df[elemental_cols].values  # 元素特征集
X_enhanced_raw = df[enhanced_cols]      # 增强特征集（可能含缺失值）

# 使用均值插补填充增强特征集中的所有缺失值（NaN）
imputer = SimpleImputer(strategy='mean')
X_enhanced = imputer.fit_transform(X_enhanced_raw)
print("Data preprocessing complete.")

# --- 步骤 2: 交叉验证与统计检验 (为图1提供数据) ---
# 初始化将在两个特征集上进行公平比较的随机森林模型
model = RandomForestClassifier(random_state=RANDOM_STATE)

# 定义交叉验证策略：10折分层交叉验证，重复3次。
# 这是一种非常严格和可靠的模型评估方法，可以减少单次划分带来的偶然性。
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)

# 在“元素特征集”上执行交叉验证，得到30个准确率分数
elemental_scores = cross_val_score(model, X_elemental, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
# 在“增强特征集”上执行交叉验证，得到另外30个准确率分数
enhanced_scores = cross_val_score(model, X_enhanced, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)

# 对两组成对的分数执行配对t检验，计算p值，以判断性能差异是否具有统计显著性
_, p_value = ttest_rel(elemental_scores, enhanced_scores)

# 将所有统计结果（均值、标准差、p值）打包到一个字典中
stats_results = {
    'mean_elemental': np.mean(elemental_scores), 'std_elemental': np.std(elemental_scores),
    'mean_enhanced': np.mean(enhanced_scores), 'std_enhanced': np.std(enhanced_scores),
    'p_value': p_value
}
# 保存统计结果，供脚本2生成图1使用
joblib.dump(stats_results, os.path.join(OUTPUT_DIR, 'statistical_results.pkl'))
print(f"Statistical evaluation complete. P-value: {p_value:.4f}")

# --- 步骤 3: 训练并保存最终的最佳模型 ---
# 根据统计检验结果（p>0.05），我们选择更简洁的“元素特征集”来训练最终模型。
# 将元素特征集数据划分为训练集（80%）和测试集（20%）。
X_train, X_test, y_train, y_test = train_test_split(
    df[elemental_cols], y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)
# 初始化最终的随机森林模型
best_model = RandomForestClassifier(random_state=RANDOM_STATE)
# 在训练集上进行模型训练
best_model.fit(X_train, y_train)
print("Best model trained.")

# --- 步骤 4: 计算并保存SHAP值 (为图4和图5提供数据) ---
print("Calculating SHAP values (this may take a moment)...")
# 创建一个SHAP解释器，用于解释基于树的模型（如随机森林）
explainer = shap.TreeExplainer(best_model)
# 在测试集上计算每个特征对每个预测的贡献（SHAP值）
shap_values = explainer.shap_values(X_test)
# 保存SHAP值，因为计算过程耗时，保存后可重复使用
joblib.dump(shap_values, os.path.join(OUTPUT_DIR, 'shap_values.pkl'))
print("SHAP values calculated and saved.")

# --- 步骤 5: 保存所有必需的“工件” (Artifacts) ---
# 将最终训练好的模型保存下来
joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'best_model.pkl'))
# 保存标签编码器，以便后续将数字标签转换回原始文本
joblib.dump(le, os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))
# 将独立的测试数据集保存下来，用于生成混淆矩阵等评估图表
joblib.dump({'X_test': X_test, 'y_test': y_test}, os.path.join(OUTPUT_DIR, 'test_data.pkl'))
print("All artifacts saved successfully.")
