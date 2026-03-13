import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# === BƯỚC 1: ĐỌC FILE CSV ===
# Thay 'Your_Kaggle_File.csv' bằng tên file bạn đã tải về
file_path = 'Student_Performance.csv' 

df = pd.read_csv(file_path)


# === BƯỚC 2: TIỀN XỬ LÝ (Nếu có cột dạng chữ) ===
# Kiểm tra nếu có cột 'Extracurricular Activities' (Yes/No) thì chuyển sang số
if 'Extracurricular Activities' in df.columns:
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# === BƯỚC 3: CHUẨN BỊ DỮ LIỆU ===
# Giả sử cột cần dự đoán là 'Performance Index'
target_col = 'Performance Index'
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === BƯỚC 4: HUẤN LUYỆN MÔ HÌNH ===
model = LinearRegression()
model.fit(X_train, y_train)

# === BƯỚC 5: DỰ ĐOÁN VÀ ĐÁNH GIÁ ===
y_pred = model.predict(X_test)

print(f"\n--- KẾT QUẢ ĐÁNH GIÁ ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# === BƯỚC 6: TRỰC QUAN HÓA VỚI SEABORN ===
plt.figure(figsize=(12, 5))

# Biểu đồ 1: Tương quan thực tế vs Dự đoán (Regplot)
plt.subplot(1, 2, 1)
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3, 's':10, 'color':'teal'}, line_kws={'color':'red'})
plt.title('Tương quan Dự đoán vs Thực tế')
plt.xlabel('Giá trị thực')
plt.ylabel('Giá trị dự đoán')

# Biểu đồ 2: Ma trận tương quan (Heatmap) - Xem biến nào quan trọng nhất
plt.subplot(1, 2, 2)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Ma trận tương quan giữa các biến')

plt.tight_layout()
plt.show()