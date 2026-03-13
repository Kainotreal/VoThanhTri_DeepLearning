import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# 1. TIỀN XỬ LÝ DỮ LIỆU
path = "data/wine.data" 
df = pd.read_csv(path, header=None)

# Tách đặc trưng (X) và nhãn (y)
# Trong wine.data: cột 0 là nhãn, cột 1-13 là đặc trưng
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# Chia tập huấn luyện và kiểm tra (tỷ lệ 70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Chuẩn hóa đặc trưng về cùng thang đo (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 2. XÂY DỰNG MÔ HÌNH LOGISTIC REGRESSION
# Cấu hình Multinomial (Softmax) cho bài toán 3 lớp
model = LogisticRegression(
    multi_class='multinomial', 
    solver='lbfgs', 
    max_iter=1000, 
    random_state=42
)
model.fit(X_train_scaled, y_train)


# 3. ĐÁNH GIÁ MÔ HÌNH
y_pred = model.predict(X_test_scaled)

print("--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 4. TRỰC QUAN HÓA DECISION BOUNDARY
# Để trực quan hóa 2D, ta chỉ chọn 2 đặc trưng đầu tiên
X_2d = X_train_scaled[:, :2] 
model_2d = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X_2d, y_train)

# Tạo lưới điểm để phủ kín không gian đặc trưng
h = .02 # Bước lưới
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Dự đoán nhãn cho từng điểm trên lưới
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Vẽ biểu đồ vùng quyết định
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, edgecolors='k', cmap='viridis')
plt.xlabel('Đặc trưng 1 (Alcohol - Scaled)')
plt.ylabel('Đặc trưng 2 (Malic Acid - Scaled)')
plt.title('Decision Boundary - Wine Classification')
plt.show()