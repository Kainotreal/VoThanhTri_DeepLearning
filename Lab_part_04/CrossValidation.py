import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
import gc
import seaborn as sns
import random
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from scipy import stats

# --- CẤU HÌNH ---
plt.style.use('seaborn-v0_8-ticks') # Cập nhật tên style mới nếu bản cũ báo lỗi

# Đọc dữ liệu (Dùng đường dẫn trực tiếp thay cho %cd)
path = r"C:\DeepLearning\Lab_part_04\medical_care.csv"
medical = pd.read_csv(path)

print(f"Dữ liệu gốc: {medical.shape}")
pd.set_option("display.max_columns", 50)

# --- 1. TRAIN/TEST SPLIT ĐƠN GIẢN ---
X_train, X_test, y_train, y_test = train_test_split(
    medical, medical.UCURNINS, test_size=0.3, random_state=random.randint(0, 1000)
)

formula = ('UCURNINS ~ UMARSTAT + USATMED + URELATE + REGION + FHOSP + FDENT + FEMER + FDOCT + ' + 
           'UIMMSTAT + UAGE + U_FTPT + U_WKSLY + U_USHRS + HOTHVAL + HRETVAL + HSSVAL + HWSVAL + UBRACE + ' + 
           'UEDUC3 + GENDER')

mod = sm.GLM.from_formula(formula=formula, data=X_train, family=sm.families.Binomial())
res = mod.fit()
preds = res.predict(X_test)
print(f"ROC AUC ban đầu: {roc_auc_score(y_test, preds):.4f}")

# --- 2. MÔ PHỎNG PHÂN PHỐI ĐIỂM SỐ (100 LẦN) ---
scores = []
for k in range(100):
    X_t, X_v, y_t, y_v = train_test_split(
        medical, medical.UCURNINS, stratify=medical.UCURNINS, 
        test_size=0.3, random_state=random.randint(0, 10000)
    )
    m = sm.GLM.from_formula(formula=formula, data=X_t, family=sm.families.Binomial()).fit()
    p = m.predict(X_v)
    scores.append(roc_auc_score(y_v, p))

df_scores = pd.DataFrame(data=scores, columns=['scores'])

# Vẽ biểu đồ phân phối
ax = sns.displot(data=df_scores.scores, kde=True)
x0, x1 = ax.ax.get_xlim()
x_pdf = np.linspace(x0, x1, len(df_scores))
y_pdf = stats.norm.pdf(x_pdf, df_scores.scores.mean(), df_scores.scores.std())
ax.ax.plot(x_pdf, y_pdf, 'r', lw=2, label='Normal PDF')
plt.legend()
plt.title("Phân phối điểm số AUC qua 100 lần lặp")
plt.show() # THÊM DÒNG NÀY ĐỂ HIỂN THỊ TRÊN VS CODE

# --- 3. KIỂM TRA CHÉO (K-FOLD) ---
print("\n--- Chạy 10-Fold Cross Validation ---")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
trainRes, valRes = [], []

for train_idx, test_idx in kf.split(medical):
    train_data = medical.iloc[train_idx]
    test_data = medical.iloc[test_idx]
    
    m = sm.GLM.from_formula(formula=formula, data=train_data, family=sm.families.Binomial()).fit()
    
    p_train = m.predict(train_data)
    p_test = m.predict(test_data)
    
    trainRes.append(roc_auc_score(train_data.UCURNINS, p_train))
    valRes.append(roc_auc_score(test_data.UCURNINS, p_test))

print(f"Trung bình Train AUC: {np.mean(trainRes):.4f}")
print(f"Trung bình Valid AUC: {np.mean(valRes):.4f}")

# --- 4. STRATIFIED K-FOLD (DÀNH CHO DỮ LIỆU LỆCH) ---
print("\n--- Chạy Stratified 5-Fold ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(medical, medical.UCURNINS):
    # Thực hiện tương tự như trên...
    pass

print("Hoàn thành quy trình đánh giá!")