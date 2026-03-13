import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load và chuẩn bị dữ liệu
iris = load_iris()
X, y = iris.data, iris.target

# 2. Chia tập Train/Test (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Chuẩn hóa dữ liệu (Feature Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Khởi tạo và huấn luyện mô hình
# Sử dụng multinomial cho bài toán 3 lớp
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Đánh giá
y_pred = model.predict(X_test_scaled)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Dự đoán cho một mẫu mới
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]]) # Ví dụ một bông hoa Setosa
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
probability = model.predict_proba(new_sample_scaled)

print(f"Kết quả dự đoán: {iris.target_names[prediction[0]]}")
print(f"Xác suất: {np.max(probability):.4f}")