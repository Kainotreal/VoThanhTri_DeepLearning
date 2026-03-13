import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. TIỀN XỬ LÝ DỮ LIỆU
print("Đang tải dữ liệu Fashion MNIST (vui lòng đợi)...")
# Tải dữ liệu từ OpenML
fashion_mnist = fetch_openml('Fashion-MNIST', version=1, parser='auto')
X, y = fashion_mnist.data, fashion_mnist.target

# Định nghĩa tên các lớp để hiển thị (từ T-shirt đến Ankle boot)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Chia tập huấn luyện và kiểm tra (60,000 train, 10,000 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y
)

# Chuẩn hóa đặc trưng bằng cách đưa về đoạn [0, 1] (thay vì StandardScaler)
# Đối với dữ liệu ảnh, chia cho 255 là phương pháp chuẩn hóa nhanh và hiệu quả nhất
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# 2. XÂY DỰNG MÔ HÌNH LOGISTIC REGRESSION
# Sử dụng solver 'saga' vì đây là bộ dữ liệu lớn (70k mẫu)
model = LogisticRegression(
    multi_class='multinomial',
    solver='saga',
    max_iter=50,      # Giới hạn số vòng lặp để chạy nhanh hơn trong bài Lab
    tol=0.1,          # Ngưỡng dừng sớm
    n_jobs=-1,        # Sử dụng tất cả nhân CPU để tăng tốc
    random_state=42
)
print("Đang huấn luyện mô hình...")
model.fit(X_train_scaled, y_train)

# 3. ĐÁNH GIÁ MÔ HÌNH
y_pred = model.predict(X_test_scaled)

print("\n--- KẾT QUẢ ĐÁNH GIÁ FASHION MNIST ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 4. TRỰC QUAN HÓA (DỰ ĐOÁN THỰC TẾ)
# Thay vì Decision Boundary (rất khó vẽ cho 784 chiều), 
# ta sẽ trực quan hóa kết quả dự đoán trên các ảnh thực tế
plt.figure(figsize=(12, 8))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    # Lấy 1 mẫu ngẫu nhiên từ tập test
    idx = np.random.randint(0, len(X_test))
    image = X_test.iloc[idx].values.reshape(28, 28)
    
    true_label = class_names[int(y_test.iloc[idx])]
    pred_label = class_names[int(y_pred[idx])]
    
    plt.imshow(image, cmap='gray')
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"T: {true_label}\nP: {pred_label}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()