import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set stdout to UTF-8 encoding to handle Vietnamese characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Tạo dữ liệu giả định
house_are = np.array([50, 70, 80, 100, 120]) # Diện tích của ngôi nhà (m2)
house_price = np.array([200, 270, 300, 370, 450]) # Giá của ngôi nhà (nghìn USD)

# Tạo dataframe
data = pd.DataFrame({
    'house_are': house_are,
    'house_price': house_price
})

data
     

# Thực hiện hồi quy tuyến tính
# Chú ý: sklearn yêu cầu reshape dữ liệu đầu vào
X = house_are.reshape(-1,1)
y = house_price

X
     

# Tạo và huấn luyện mô hình
model = LinearRegression()
model.fit(X, y)
     

print("Điểm cắt (intercept): " , model.intercept_)
print("Hệ số góc (slope): ", model.coef_[0])
print("R-squared:", model.score(X, y))
     

# Vẽ biểu đồ
plt.figure(figsize=(10,6))
plt.scatter(house_are, house_price, color='red', label='Dữ liệu gốc')
plt.plot(house_are, model.predict(X), color='blue', linewidth=2, label='Đường hồi quy tuyến tính')
plt.xlabel('Diện tích (m2)')
plt.ylabel('Giá nhà (nghìn USD)')
plt.title('Hồi quy tuyến tính cho giá nhà')
plt.legend()
plt.show()
     

# Sử dụng mô hình để dự đoán
new_house_area = np.array([85, 110, 150]).reshape(-1,1)
predicted_prices = model.predict(new_house_area)

print("\nDự đoán giá nhà cho diện tích 85m2, 110m2 và 150m2:")
for area, price in zip(new_house_area.flatten(), predicted_prices):
   print(f"Diện tích {area} m²: {price:.2f} nghìn USD")