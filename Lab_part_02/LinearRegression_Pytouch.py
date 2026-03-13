import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# === BƯỚC 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ===
# Hãy đảm bảo file 'Student_Performance.csv' nằm cùng thư mục
try:
    df = pd.read_csv('Student_Performance.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file CSV. Vui lòng kiểm tra lại!")
    exit()

# Chuyển đổi dữ liệu chữ sang số
if 'Extracurricular Activities' in df.columns:
    df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Tách đặc trưng và mục tiêu
X = df.drop('Performance Index', axis=1).values
y = df['Performance Index'].values.reshape(-1, 1)

# Chuẩn hóa dữ liệu (Bắt buộc đối với Deep Learning/PyTorch)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Chia tập huấn luyện và tập kiểm tra (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Chuyển đổi sang PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# === BƯỚC 2: XÂY DỰNG MÔ HÌNH ===
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(X_train_t.shape[1])

# === BƯỚC 3: CÀI ĐẶT HUẤN LUYỆN ===
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam thường tốt hơn SGD cho dữ liệu thực tế
losses = []

# === BƯỚC 4: VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP) ===
epochs = 100
print("--- Đang bắt đầu huấn luyện ---")
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    
    # Backward pass và tối ưu
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# === BƯỚC 5: DỰ ĐOÁN VÀ ĐÁNH GIÁ ===
model.eval()
with torch.no_grad():
    y_pred_t = model(X_test_t)
    
    # Nghịch đảo chuẩn hóa để về giá trị điểm thực tế (0-100)
    y_test_actual = scaler_y.inverse_transform(y_test_t.numpy())
    y_pred_actual = scaler_y.inverse_transform(y_pred_t.numpy())
    
    r2 = r2_score(y_test_actual, y_pred_actual)
    print(f"\n--- ĐÁNH GIÁ MÔ HÌNH ---")
    print(f"R2 Score: {r2:.4f}")

# === BƯỚC 6: TRỰC QUAN HÓA CHUYÊN NGHIỆP ===
sns.set_theme(style="darkgrid")
plt.figure(figsize=(18, 5))

# Biểu đồ 1: Quá trình giảm Loss
plt.subplot(1, 3, 1)
plt.plot(losses, color='blue', lw=2)
plt.title('Đường cong Huấn luyện (Loss)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

# Biểu đồ 2: Tương quan Thực tế vs Dự đoán
plt.subplot(1, 3, 2)
sns.scatterplot(x=y_test_actual.flatten(), y=y_pred_actual.flatten(), alpha=0.3, color='teal')
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
plt.title('Thực tế vs Dự đoán')
plt.xlabel('Giá trị thực')
plt.ylabel('Giá trị dự đoán')

# Biểu đồ 3: Phân phối sai số
plt.subplot(1, 3, 3)
residuals = y_test_actual - y_pred_actual
sns.histplot(residuals, kde=True, color='purple')
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Phân phối sai số (Residuals)')

plt.tight_layout()
plt.show()

# === BƯỚC 7: DỰ ĐOÁN DỮ LIỆU MỚI ===
def predict_new(features):
    model.eval()
    with torch.no_grad():
        features_scaled = scaler_X.transform(np.array([features]))
        features_t = torch.tensor(features_scaled, dtype=torch.float32)
        pred_scaled = model(features_t)
        pred_actual = scaler_y.inverse_transform(pred_scaled.numpy())
        return pred_actual[0][0]

# Thử nghiệm với dữ liệu mới
# [Hours Studied, Previous Scores, Extracurricular Activities, Sleep Hours, Sample Question Papers Practiced]
new_student = [7, 90, 1, 8, 4]
result = predict_new(new_student)
print(f"\n--- DỰ ĐOÁN SINH VIÊN MỚI ---")
print(f"Với đầu vào {new_student} => Điểm dự kiến: {result:.2f}")