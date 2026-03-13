import pandas as pd
import numpy as np

# Thiết lập seed để đảm bảo bạn chạy lại sẽ ra kết quả giống hệt tôi
np.random.seed(42)

# 1. Tạo 500 mẫu diện tích (từ 30m2 đến 250m2)
n_samples = 10000
square_meters = np.random.uniform(30, 250, n_samples)

# 2. Tạo giá nhà dựa trên quy luật thực tế: 
# Giá = 500 (triệu) + 35 * diện tích + nhiễu ngẫu nhiên
# (Mỗi m2 giá khoảng 35 triệu VNĐ)
noise = np.random.normal(0, 100, n_samples) # Sai số khoảng 100 triệu
prices = 500 + (35 * square_meters) + noise

# 3. Đóng gói vào DataFrame và lưu file
df = pd.DataFrame({
    'Square_Footage': square_meters.round(2),
    'Price': prices.round(2)
})

df.to_csv('housing_data.csv', index=False)

print("Đã tạo thành công file 'housing_data.csv' với 500 dòng!")
print(df.head()) # Xem thử 5 dòng đầu