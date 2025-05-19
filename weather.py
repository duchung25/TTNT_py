import numpy as np
from sklearn.naive_bayes import GaussianNB
import pickle

# Dữ liệu ví dụ: [Nhiệt độ (C), Độ ẩm (%), Gió (0: yếu, 1: mạnh)]
X = np.array([
    [30, 80, 0],
    [25, 60, 1],
    [28, 90, 0],
    [23, 70, 1],
    [35, 40, 0],
    [31, 60, 1]
])
# Nhãn: 1 = Mưa, 0 = Không mưa
y = np.array([1, 0, 1, 0, 0, 0])

# Huấn luyện mô hình
model = GaussianNB()
model.fit(X, y)

# Lưu mô hình
with open("weather_nb.pkl", "wb") as f:
    pickle.dump(model, f)