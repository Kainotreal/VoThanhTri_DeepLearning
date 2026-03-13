import time
import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Tải dữ liệu MNIST (784 đặc trưng)
print("Đang tải dữ liệu MNIST...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# 2. Tiền xử lý: Chuẩn hóa về đoạn [0, 1] và chia dữ liệu
X_normalized = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=10000, train_size=60000, random_state=42
)

# 3. Cấu hình mô hình cho dữ liệu lớn
# Dùng solver 'saga' và tận dụng đa nhân CPU (n_jobs=-1)
model = LogisticRegression(
    solver='saga', 
    multi_class='multinomial', 
    max_iter=100, 
    tol=0.1, 
    n_jobs=-1, 
    random_state=42,
    verbose=1
)

# 4. Huấn luyện và đo thời gian
start_time = time.time()
model.fit(X_train, y_train)
print(f"Huấn luyện xong trong: {time.time() - start_time:.2f} giây")

# 5. Đánh giá chi tiết
y_pred = model.predict(X_test)
print("\nKết quả trên tập Test:")
print(classification_report(y_test, y_pred))

# 6. Tìm chữ số khó nhận diện nhất (Accuracy thấp nhất)
cm = confusion_matrix(y_test, y_pred)
accuracy_per_digit = cm.diagonal() / cm.sum(axis=1)
worst_digit = np.argmin(accuracy_per_digit)
print(f"Chữ số khó nhận diện nhất là {worst_digit} với độ chính xác {accuracy_per_digit[worst_digit]:.4f}")

# 7. Lưu mô hình để tái sử dụng
joblib.dump(model, 'mnist_logistic_model.pkl')