import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import threading

model = tf.keras.models.load_model('/Users/benny/Desktop/Ben Study/HỌC KÌ 7/xử lí ảnh và thị giác máy tính/newpj/traffic_sign_recognition/traffic_classifier2.h5')

class MediaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("App nhận diện biển báo")
        self.root.geometry('1200x900')
        self.root.config(bg="#ffe4e1")  # Màu nền dịu

        # Khung tiêu đề với màu sắc hài hoà
        self.title_frame = tk.Frame(root, bg='#ffc0cb', bd=5, relief="raised")
        self.title_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        self.title_label = tk.Label(self.title_frame, text="[010100086902] - Xử lý ảnh và thị giác máy tính - Nhóm 9", font=("Arial", 20, "bold"), fg="black", bg='#0044cc')
        self.title_label.pack(pady=10, padx=300)

        # Các khung trái, giữa, phải
        self.left_frame = tk.Frame(root, bg='#ffc0cb', bd=3, relief="groove")
        self.left_frame.grid(row=1, column=0, padx=10, pady=10, sticky="n")

        self.middle_frame = tk.Frame(root, bg='#ffc0cb', bd=3, relief="groove")
        self.middle_frame.grid(row=1, column=1, padx=10, pady=10, sticky="n")

        self.right_frame = tk.Frame(root, bg='#ffc0cb', bd=3, relief="groove")
        self.right_frame.grid(row=1, column=2, padx=10, pady=10, sticky="n")

        self.image_label = None
        self.video_label = None
        self.camera_label = None
        self.camera_on = False
        self.cap = None

        self.class_labels = {0: "Dấu hiệu 1", 1: "Dấu hiệu 2", 2: "Dấu hiệu 3"}

        # Label hiển thị kết quả dự đoán, đặt giữa màn hình
        self.prediction_label = tk.Label(self.root, text="", font=("Arial", 16, "italic"), bg="#ffc0cb", fg="#ff3333")
        self.prediction_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        # Nút thêm hình ảnh với màu sắc
        self.add_image_button = tk.Button(self.left_frame, text="Thêm Hình Ảnh", command=self.add_image, width=20)
        self.add_image_button.pack(pady=10, anchor="w")

        self.add_video_button = tk.Button(self.left_frame, text="Thêm Video", command=self.add_video, width=20)
        self.add_video_button.pack(pady=10, anchor="w")

        self.toggle_camera_button = tk.Button(self.left_frame, text="Bật/Tắt Camera", command=self.toggle_camera, width=20)
        self.toggle_camera_button.pack(pady=10, anchor="w")

        self.clear_all_button = tk.Button(self.left_frame, text="Xóa Tất Cả", command=self.clear_all, width=20)
        self.clear_all_button.pack(pady=10, anchor="w")


    def add_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            img = Image.open(file_path)
            img = img.resize((300, 300), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(img)

            if self.image_label is None:
                self.image_label = tk.Label(self.right_frame, image=img)
                self.image_label.image = img
                self.image_label.pack(pady=10)
            else:
                self.image_label.config(image=img)
                self.image_label.image = img

            # Dự đoán với mô hình Keras
            threading.Thread(target=self.predict_image, args=(file_path,)).start()

    def getClassName(self, classNo):
        class_names = [
            'Giới hạn tốc độ 20 km/h', 'Giới hạn tốc độ 30 km/h', 'Giới hạn tốc độ 50 km/h',
            'Giới hạn tốc độ 60 km/h', 'Giới hạn tốc độ 70 km/h', 'Giới hạn tốc độ 80 km/h',
            'Kết thúc giới hạn tốc độ 80 km/h', 'Giới hạn tốc độ 100 km/h', 'Giới hạn tốc độ 120 km/h',
            'Cấm vượt', 'Cấm vượt đối với xe trên 3,5 tấn', 'Được ưu tiên tại giao lộ kế tiếp',
            'Đường ưu tiên', 'Nhường đường', 'Dừng lại', 'Cấm xe', 
            'Cấm xe trên 3,5 tấn', 'Cấm vào', 'Cảnh báo chung', 'Đường cong nguy hiểm bên trái', 
            'Đường cong nguy hiểm bên phải', 'Đường có hai khúc cua', 'Đường xấu', 
            'Đường trơn', 'Đường hẹp bên phải', 'Đang thi công', 'Tín hiệu giao thông', 
            'Người đi bộ', 'Trẻ em băng qua đường', 'Xe đạp băng qua đường', 
            'Cẩn thận băng tuyết', 'Động vật hoang dã băng qua đường', 
            'Kết thúc tất cả giới hạn tốc độ và vượt', 'Rẽ phải phía trước', 'Rẽ trái phía trước', 
            'Chỉ đi thẳng', 'Đi thẳng hoặc rẽ phải', 'Đi thẳng hoặc rẽ trái', 
            'Đi về bên phải', 'Đi về bên trái', 'Vòng xuyến bắt buộc', 
            'Kết thúc cấm vượt', 'Kết thúc cấm vượt với xe trên 3,5 tấn'
        ]
        return class_names[classNo] if classNo < len(class_names) else "Unknown"

    def predict_image(self, file_path):
        try:
            data = []
            # Đọc hình ảnh
            image_path = Image.open(file_path)

            image_path = image_path.convert('RGB')

            # Đảm bảo kích thước hình ảnh phù hợp với mô hình
            image = image_path.resize([30, 30])  # Gán kích thước cho biến image
            data.append(np.array(image))
            image_array = np.array(data)

            print(f'Input shape for prediction: {image_array.shape}')  # Kiểm tra kích thước đầu vào

            # Dự đoán lớp
            predictions = model.predict(image_array)
            pred = np.argmax(predictions, axis=-1)
            confidence = np.max(predictions)  # Lấy độ tự tin từ dự đoán
            class_name = self.getClassName(pred[0])

            # Hiển thị kết quả
            messagebox.showinfo("Prediction", f"Dự đoán: {class_name}\nĐộ tự tin: {confidence:.2f}")

            if confidence >= 0.85:
                self.prediction_label.config(text=f"{class_name}\nĐộ tự tin: {confidence:.2f}")
            else:
                self.prediction_label.config(text="Không đủ độ tự tin để dự đoán.")
        except Exception as e:
            messagebox.showerror("Error", f"Có lỗi xảy ra trong quá trình dự đoán: {str(e)}")

    # def draw_prediction(self, file_path, predicted_name):
    #     img = cv2.imread(file_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #     # Vẽ ô vuông màu xanh quanh biển báo dự đoán
    #     img = Image.fromarray(img)
    #     img = img.resize((300, 300), Image.Resampling.LANCZOS)  # Resize để hiển thị
    #     img_tk = ImageTk.PhotoImage(image=img)

    #     if self.image_label is None:
    #         self.image_label = tk.Label(self.right_frame, image=img_tk)
    #         self.image_label.image = img_tk
    #         self.image_label.pack(pady=10)
    #     else:
    #         self.image_label.config(image=img_tk)
    #         self.image_label.image = img_tk

    def add_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.play_video()

    def play_video(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kernel hình tròn 5x5
                frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
                self.predict_video_frame(frame)  # Gọi hàm dự đoán cho từng khung hình

                
                # Không cần hiển thị frame ở kích thước 300x300, chỉ cần resize về 30x30 cho mô hình
                img = Image.fromarray(frame)
                img = img.resize((300, 300), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(image=img)

                if self.video_label is None:
                    self.video_label = tk.Label(self.middle_frame, image=img_tk)
                    self.video_label.image = img_tk
                    self.video_label.pack(pady=10)
                else:
                    self.video_label.config(image=img_tk)
                    self.video_label.image = img_tk

                self.root.after(30, self.play_video)
            else:
                self.cap.release()

    def predict_video_frame(self, frame):
        if model is not None:
            try:
                data = []
                img_resized = cv2.resize(frame, (30, 30))  # Resize thành 30x30
                data.append(np.array(img_resized))
                image_array = np.array(data)

                print(f'Input shape for prediction: {image_array.shape}')

                # Dự đoán lớp
                predictions = model.predict(image_array)
                pred = np.argmax(predictions, axis=-1)
                confidence = np.max(predictions)  # Lấy độ tự tin từ dự đoán
                class_name = self.getClassName(pred[0])

                # Cập nhật label dự đoán chỉ khi độ tự tin >= 95%
                if confidence >= 0.85:
                    self.prediction_label.config(text=f"{class_name}\nĐộ tự tin: {confidence:.2f}")
                else:
                    self.prediction_label.config(text="Không đủ độ tự tin để dự đoán.")

            except Exception as e:
                print(f"Error in predicting video frame: {e}")

    def toggle_camera(self):
        if self.camera_on:
            self.camera_on = False
            if self.cap:
                self.cap.release()
            if self.camera_label:
                self.camera_label.destroy()
                self.camera_label = None
            self.prediction_label.config(text="")  # Xóa thông tin dự đoán
        else:
            self.camera_on = True
            self.cap = cv2.VideoCapture(0)  # Mở camera
            self.play_camera()

    def play_camera(self):
        if self.camera_on and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Áp dụng kernel hình tròn để làm mịn hình ảnh
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kernel hình tròn 5x5
                frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)  # Sử dụng MORPH_CLOSE để làm mịn

                self.predict_video_frame(frame)  # Gọi hàm dự đoán cho từng khung hình
                
                # Không cần hiển thị frame ở kích thước 300x300, chỉ cần resize về 30x30 cho mô hình
                img = Image.fromarray(frame)
                img = img.resize((300, 150), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(image=img)

                if self.camera_label is None:
                    self.camera_label = tk.Label(self.middle_frame, image=img_tk)
                    self.camera_label.image = img_tk
                    self.camera_label.pack(pady=10)
                else:
                    self.camera_label.config(image=img_tk)
                    self.camera_label.image = img_tk

                self.root.after(30, self.play_camera)

    def clear_all(self):
        if self.image_label:
            self.image_label.destroy()
            self.image_label = None
        if self.video_label:
            self.video_label.destroy()
            self.video_label = None
        if self.camera_label:
            self.camera_label.destroy()
            self.camera_label = None
        self.prediction_label.config(text="")  # Xóa thông tin dự đoán

if __name__ == "__main__":
    root = tk.Tk()
    app = MediaApp(root)
    root.mainloop()