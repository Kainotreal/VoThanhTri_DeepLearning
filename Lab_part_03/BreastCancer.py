import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 1. TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
# Load dữ liệu trực tiếp từ thư viện Sklearn
data = load_breast_cancer()
X, y = data.data, data.target

# Chia tập huấn luyện và kiểm tra (tỷ lệ 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa đặc trưng (Scaling)
# Rất quan trọng vì các đặc trưng như 'area' và 'smoothness' có thang đo lệch nhau rất lớn
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. XÂY DỰNG MÔ HÌNH LOGISTIC REGRESSION
# ==========================================
# Đối với phân loại nhị phân, ta có thể dùng mặc định hoặc cấu hình 'ovr'
model = LogisticRegression(
    solver='lbfgs', 
    max_iter=1000, 
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ==========================================
# 3. ĐÁNH GIÁ MÔ HÌNH
# ==========================================
y_pred = model.predict(X_test_scaled)

print("--- KẾT QUẢ ĐÁNH GIÁ BREAST CANCER ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# ==========================================
# 4. TRỰC QUAN HÓA DECISION BOUNDARY
# ==========================================
# Chọn 2 đặc trưng quan trọng nhất để vẽ 2D (ví dụ: mean radius và mean texture)
X_2d = X_train_scaled[:, :2]
model_2d = LogisticRegression().fit(X_2d, y_train)

# Tạo lưới điểm
h = .02
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Dự đoán trên lưới
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu') # Đỏ cho Ác tính, Xanh cho Lành tính
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, edgecolors='k', cmap='RdBu')
plt.xlabel('Mean Radius (Scaled)')
plt.ylabel('Mean Texture (Scaled)')
plt.title('Decision Boundary - Breast Cancer (Binary Classification)')
plt.show()