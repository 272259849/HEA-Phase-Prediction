# 1_train_and_evaluate.py进行模型训练和统计评估，这是经过前期调试、整合精简后的代码。
# 该脚本用于训练随机森林模型并进行统计评估，对比元素特征集和增强特征集的模型性能，最佳模型为best_model.pkl



import pandas as pd
import numpy as np
import joblib
import os
import shap
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_rel


RANDOM_STATE = 42
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
except NameError:
    project_root = os.getcwd()
OUTPUT_DIR = os.path.join(project_root, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Data Loading and Preprocessing
df = pd.read_excel(os.path.join(project_root, 'HEA_DataSet.xlsx'))

elemental_cols = ['Al', 'Co', 'Cr', 'Fe', 'Ni', 'Cu', 'Mn', 'Ti', 'V', 'Nb', 'Mo', 'Zr', 'Hf', 'Ta', 'W', 'C', 'Mg', 'Zn', 'Si', 'Re', 'N', 'Sc', 'Li', 'Sn', 'Be']
physical_cols = ['VEC', 'dSmix', 'dHmix', 'Atom.Size.Diff', 'Elect.Diff', 'Density_calc', 'Tm']
enhanced_cols = elemental_cols + physical_cols
target_col = 'Phases'

le = LabelEncoder()
y_encoded = le.fit_transform(df[target_col])

X_elemental = df[elemental_cols].values
X_enhanced_raw = df[enhanced_cols]
imputer = SimpleImputer(strategy='mean')
X_enhanced = imputer.fit_transform(X_enhanced_raw)
print("Data preprocessing complete.")

#Cross-Validation and Statistical Test
model = RandomForestClassifier(random_state=RANDOM_STATE)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)

elemental_scores = cross_val_score(model, X_elemental, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
enhanced_scores = cross_val_score(model, X_enhanced, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
_, p_value = ttest_rel(elemental_scores, enhanced_scores)

stats_results = {
    'mean_elemental': np.mean(elemental_scores), 'std_elemental': np.std(elemental_scores),
    'mean_enhanced': np.mean(enhanced_scores), 'std_enhanced': np.std(enhanced_scores),
    'p_value': p_value
}
joblib.dump(stats_results, os.path.join(OUTPUT_DIR, 'statistical_results.pkl'))
print(f"Statistical evaluation complete. P-value: {p_value:.4f}")

#Train and Save Final Model
X_train, X_test, y_train, y_test = train_test_split(
    df[elemental_cols], y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)
best_model = RandomForestClassifier(random_state=RANDOM_STATE)
best_model.fit(X_train, y_train)
print("Best model trained.")

#Calculate and Save SHAP Values (New Step)
print("Calculating SHAP values (this may take a moment)...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
joblib.dump(shap_values, os.path.join(OUTPUT_DIR, 'shap_values.pkl'))
print("SHAP values calculated and saved.")

#5. Save All Artifacts
joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'best_model.pkl'))
joblib.dump(le, os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))
joblib.dump({'X_test': X_test, 'y_test': y_test}, os.path.join(OUTPUT_DIR, 'test_data.pkl'))
print("All artifacts saved successfully.")
