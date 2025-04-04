import tkinter as tk
from tkinter import filedialog, messagebox, Canvas
import cv2
import numpy as np
from PIL import Image, ImageTk
import pytesseract
from LicensePlateRecognizer import LicensePlateRecognizer  # Nhớ import lớp LicensePlateRecognizer của bạn

# Cấu hình Tesseract OCR (đường dẫn có thể khác nhau tùy hệ thống)
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện biển số xe")
        self.root.geometry("1000x600")
        self.root.configure(bg="#ECF0F1")

        # Khởi tạo đối tượng LicensePlateRecognizer
        self.plate_recognizer = LicensePlateRecognizer(languages=['en', 'vi'])

        # Thiết lập lưới co giãn
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)

        # Tiêu đề ứng dụng
        self.title_label = tk.Label(root, text="NHẬN DIỆN BIỂN SỐ XE", font=("Arial", 18, "bold"), bg="#ECF0F1", fg="#2C3E50", bd=2, relief="solid")
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="ew")

        # Khung chứa hình ảnh hoặc video
        self.image_frame = tk.Frame(root, bg="#BDC3C7", relief=tk.RAISED, bd=5)
        self.image_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)

        self.image_label = tk.Label(self.image_frame, bg="#95A5A6", bd=2, relief="sunken")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # Khung điều khiển
        self.control_frame = tk.Frame(root, bg="#ECF0F1")
        self.control_frame.grid(row=1, column=1, padx=20, pady=20, sticky="ns")

        # Thông báo trạng thái
        self.status_label = tk.Label(self.control_frame, text="Vui lòng bật camera để sử dụng", font=("Arial", 10), fg="red", bg="#ECF0F1")
        self.status_label.pack(pady=10)

        # Khung chứa nút bật/tắt camera
        self.camera_control_frame = tk.Frame(self.control_frame, bg="#ECF0F1")
        self.camera_control_frame.pack(pady=10)

        self.start_camera_button = tk.Button(self.camera_control_frame, text="Bắt đầu Camera", command=self.start_camera, font=("Arial", 12), bg="#3498DB", fg="white", width=15, height=2, bd=2, relief="groove")
        self.start_camera_button.grid(row=0, column=0, padx=5)

        self.stop_camera_button = tk.Button(self.camera_control_frame, text="Tắt Camera", command=self.stop_camera, font=("Arial", 12), bg="#E74C3C", fg="white", width=15, height=2, bd=2, relief="groove")
        self.stop_camera_button.grid(row=0, column=1, padx=5)

        # Các nút chức năng
        self.capture_button = self.create_button("Chụp Hình", self.capture_image, "#2ECC71")
        self.record_button = self.create_button("Quay Video", self.record_video, "#9B59B6")
        self.upload_button = self.create_button("Tải Lên", self.upload_image, "#F1C40F")
        self.recognize_button = self.create_button("Nhận diện Biển Số", self.recognize_license_plate, "#E74C3C")

        self.camera = None
        self.image = None
        self.recording = False

    def create_button(self, text, command, color):
        button = tk.Button(self.control_frame, text=text, command=command, font=("Arial", 12), bg=color, fg="white", width=20, height=2, bd=2, relief="groove")
        button.pack(pady=5, fill="x")  # Co giãn theo chiều ngang
        return button

    def start_camera(self):
        if not self.camera:
            self.status_label.config(text="Đang bật camera...")
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở camera. Vui lòng kiểm tra lại.")
                self.camera = None
                return
            self.show_frame()
        else:
            self.status_label.config(text="Camera đã bật!")

    def stop_camera(self):
        if self.camera:
            self.camera.release()
            self.camera = None
            self.status_label.config(text="Đã tắt camera")
            self.image_label.config(image='')  # Xóa hình ảnh
        else:
            self.status_label.config(text="Camera chưa bật!")

    def show_frame(self):
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.image = Image.fromarray(frame)

                # Cập nhật kích thước ảnh theo khung hình
                self.image = self.image.resize((self.image_label.winfo_width(), self.image_label.winfo_height()), Image.Resampling.LANCZOS)

                self.photo = ImageTk.PhotoImage(image=self.image)
                self.image_label.config(image=self.photo)
                self.image_label.image = self.photo
            self.root.after(10, self.show_frame)

    def capture_image(self):
        if not self.camera:
            self.status_label.config(text="Vui lòng bật camera trước!")
            return
        self.status_label.config(text="")
        ret, frame = self.camera.read()
        if ret:
            self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=self.image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

    def record_video(self):
        if not self.camera:
            self.status_label.config(text="Vui lòng bật camera trước!")
            return
        self.status_label.config(text="")
        self.recording = not self.recording
        if self.recording:
            self.status_label.config(text="Bắt đầu quay video...")
        else:
            self.status_label.config(text="Dừng quay video.")

    def upload_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if filename:
            self.image = Image.open(filename)

            # Cập nhật kích thước ảnh theo khung hình
            self.image = self.image.resize((self.image_label.winfo_width(), self.image_label.winfo_height()), Image.Resampling.LANCZOS)

            self.photo = ImageTk.PhotoImage(image=self.image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

    def recognize_license_plate(self):
        if not self.image:
            messagebox.showerror("Lỗi", "Vui lòng chọn hoặc chụp ảnh trước!")
            return
        
        # Chuyển đổi ảnh PIL sang OpenCV
        open_cv_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)

        # Gọi hàm nhận diện biển số xe
        processed_frame, license_plate_texts = self.plate_recognizer.detect_license_plate_and_text(open_cv_image)

        # Hiển thị ảnh đã xử lý
        # Chuyển ảnh OpenCV (BGR) sang PIL (RGB)
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        processed_image = Image.fromarray(processed_frame_rgb)

        # Resize ảnh để vừa với giao diện
        processed_image = processed_image.resize((self.image_label.winfo_width(), self.image_label.winfo_height()), Image.Resampling.LANCZOS)

        # Hiển thị ảnh trên Tkinter
        self.photo = ImageTk.PhotoImage(image=processed_image)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

        if license_plate_texts:
            messagebox.showinfo("Kết quả nhận diện", f"Biển số xe: {', '.join(license_plate_texts)}")
            cv2.imwrite("detected_license_plate.jpg", processed_frame)
            messagebox.showinfo("Thông báo", "Đã lưu ảnh biển số vào detected_license_plate.jpg")
        else:
            messagebox.showinfo("Kết quả nhận diện", "Không nhận diện được biển số xe.")


if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()
