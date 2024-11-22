import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score
import os

# Tải mô hình đã lưu
model = tf.keras.models.load_model('traffic_classifier.h5')

# Nhập tập dữ liệu kiểm tra
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

# Lấy hình ảnh
with tf.device('/GPU:0'):
    for img in imgs:
        # Đảm bảo rằng đường dẫn đến ảnh được tạo đúng
        image_path = os.path.join('Dataset', img)  # Điều chỉnh ở đây
        image = Image.open(image_path)  
        image = image.resize([30, 30])
        data.append(np.array(image))

X_test = np.array(data)

# Dự đoán với mô hình
with tf.device('/GPU:0'):
    pred = np.argmax(model.predict(X_test), axis=-1)

# Tính độ chính xác với dữ liệu kiểm tra
print("Accuracy:", accuracy_score(labels, pred))

# In ra nhãn của 10 bức hình đầu tiên
print("Predicted labels for the first 10 images:", pred[:10])
