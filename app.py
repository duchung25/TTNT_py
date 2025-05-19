from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load mô hình đã huấn luyện
with open("weather_nb.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Lấy dữ liệu đầu vào: nhiệt độ, độ ẩm, gió
    temperature = data.get("temperature")
    humidity = data.get("humidity")
    wind = data.get("wind")  # 0: yếu, 1: mạnh
    X = np.array([[temperature, humidity, wind]])
    y_pred = model.predict(X)[0]
    result = "Mưa" if y_pred == 1 else "Không mưa"
    return jsonify({"dự_báo": result})

if __name__ == "__main__":
    app.run(debug=True)