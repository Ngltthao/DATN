import tkinter as tk
from tkinter import filedialog, messagebox, Canvas
import cv2
import numpy as np
from PIL import Image, ImageTk
import pytesseract
from LicensePlateRecognizer import LicensePlateRecognizer  # Nhớ import lớp LicensePlateRecognizer của bạn
from tkinter.simpledialog import askinteger
from tkinter import Scale, HORIZONTAL
import threading
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

        # Thêm thanh trượt phóng to
        self.zoom_slider = Scale(self.control_frame, from_=1, to=3, orient=HORIZONTAL, resolution=0.01, label="Phóng To", command=self.update_zoom, length=300)
        self.zoom_slider.set(1)  # Thiết lập mức độ phóng to ban đầu
        self.zoom_slider.pack(pady=10)

        # Khung chứa nút làm nét/ đặt lại
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=2, column=0, pady=10)  # Đặt row và column theo yêu cầu của bạn

        btn_sharpen = tk.Button(button_frame, text="Làm nét ảnh", command=self.sharpen_image)
        btn_sharpen.pack(side=tk.LEFT, padx=5)

        btn_reset = tk.Button(button_frame, text="Đặt lại ảnh", command=self.reset_image)
        btn_reset.pack(side=tk.LEFT, padx=5)

        # Nút chứa nhận diện biển số 
        self.recognize_button = self.create_button("Nhận diện Biển Số", self.recognize_license_plate, "#E74C3C")

        self.camera = None
        self.image = None
        self.recording = False
        self.zoom_factor = 1  # Lưu trữ tỉ lệ phóng to hiện tại
        self.offset_x = 0  # Độ lệch của ảnh theo chiều ngang
        self.offset_y = 0  # Độ lệch của ảnh theo chiều dọc
        self.image_position = (0, 0)  # Lưu trữ vị trí ảnh hiện tại (dùng cho việc di chuyển ảnh)
        self.is_dragging = False  # Biến kiểm tra xem người dùng có đang kéo ảnh không

        # Sự kiện để di chuyển ảnh khi nhấn đúp chuột
        self.image_label.bind("<Button-1>", self.on_image_click)  # Nhấp chuột
        self.image_label.bind("<B1-Motion>", self.on_image_drag)  # Di chuyển chuột khi nhấn giữ
        self.image_label.bind("<Double-1>", self.on_double_click)  # Đúp chuột

    def update_zoom(self, val):
        self.zoom_factor = float(val)  # Cập nhật tỉ lệ phóng to từ thanh trượt
        self.display_image()

    def on_image_click(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self.is_dragging = True

    def on_image_drag(self, event):
        if self.is_dragging:
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            self.offset_x += dx
            self.offset_y += dy
            self.image_position = (self.offset_x, self.offset_y)
            self.last_x = event.x
            self.last_y = event.y
            self.display_image()

    def on_double_click(self, event):
        """ Kích hoạt chế độ kéo ảnh khi đúp chuột """
        self.is_dragging = True  # Bắt đầu kéo ảnh
        self.offset_x = event.x
        self.offset_y = event.y

    def display_image(self):
        if self.image:
            # Tính toán lại kích thước ảnh dựa trên tỉ lệ phóng to
            width, height = self.image.size
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)
            resized_image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Cắt ảnh khi cần thiết nếu quá lớn so với vùng hiển thị
            resized_image = resized_image.crop((
                self.image_position[0], self.image_position[1],
                self.image_position[0] + new_width, self.image_position[1] + new_height
            ))

            # Hiển thị ảnh sau khi phóng to và di chuyển
            self.photo = ImageTk.PhotoImage(image=resized_image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

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
            # Bắt đầu hiển thị khung hình từ camera liên tục
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
                # Chuyển đổi khung hình từ BGR sang RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.image = Image.fromarray(frame_rgb)
                # Chạy nhận diện biển số xe
                processed_frame, license_plate_texts = self.plate_recognizer.detect_license_plate_and_text(frame)

                # Hiển thị ảnh đã xử lý với biển số nhận diện được
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                processed_image = Image.fromarray(processed_frame_rgb)

                # Resize ảnh để vừa với giao diện
                processed_image = processed_image.resize((self.image_label.winfo_width(), self.image_label.winfo_height()), Image.Resampling.LANCZOS)

                # Cập nhật ảnh lên giao diện
                self.photo = ImageTk.PhotoImage(image=processed_image)
                self.image_label.config(image=self.photo)
                self.image_label.image = self.photo

                # Hiển thị kết quả nhận diện biển số xe
                if license_plate_texts:
                    self.status_label.config(text=f"Biển số nhận diện: {', '.join(license_plate_texts)}")
                else:
                    self.status_label.config(text="Không nhận diện được biển số.")

            # Tiếp tục cập nhật khung hình sau 1500ms (1.5 giây)
            self.root.after(100, self.show_frame)

    def capture_image(self):
        if not self.camera:
            self.status_label.config(text="Vui lòng bật camera trước!")
            return
        
        # Chụp ảnh từ camera
        ret, frame = self.camera.read()
        if ret:
            # Dừng camera sau khi chụp
            self.camera.release()
            self.camera = None

            # Chuyển ảnh chụp thành ảnh PIL để hiển thị
            self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=self.image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

            # Chạy nhận diện biển số xe trên hình ảnh đã chụp
            processed_frame, license_plate_texts = self.plate_recognizer.detect_license_plate_and_text(frame)

            # Hiển thị kết quả nhận diện trên màn hình
            if license_plate_texts:
                self.status_label.config(text=f"Biển số nhận diện: {', '.join(license_plate_texts)}")
            else:
                self.status_label.config(text="Không nhận diện được biển số.")

    def record_video(self):
        if not self.camera:
            self.status_label.config(text="Vui lòng bật camera trước!")
            return

        self.recording = not self.recording

        if self.recording:
            self.status_label.config(text="Bắt đầu quay video...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (640, 480))
            self.recording_thread = threading.Thread(target=self._record_loop)
            self.recording_thread.start()
        else:
            self.status_label.config(text="Dừng quay video.")
            if hasattr(self, 'video_writer'):
                self.video_writer.release()

    def _record_loop(self):
        while self.recording and self.camera:
            ret, frame = self.camera.read()
            if ret:
                self.video_writer.write(frame)
                # Không nên dùng cv2.imshow trong app Tkinter. Có thể bỏ dòng này.

    def upload_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if filename:
            self.image = Image.open(filename)

            # Cập nhật kích thước ảnh theo khung hình
            self.image = self.image.resize((self.image_label.winfo_width(), self.image_label.winfo_height()), Image.Resampling.LANCZOS)
            
            # Lưu ảnh gốc
            self.original_image = self.image.copy()

            # Hiển thị ảnh
            self.photo = ImageTk.PhotoImage(image=self.image)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

            # Lưu ảnh gốc trước khi thực hiện các thao tác khác
            self.original_image = self.image.copy()

    def sharpen_image(self): 
        if not self.image:
            messagebox.showerror("Lỗi", "Vui lòng chọn hoặc chụp ảnh trước!")
            return

        # Lưu ảnh gốc trước khi làm nét (nếu chưa lưu)
        if not hasattr(self, "original_image") or self.original_image is None:
            self.original_image = self.image.copy()

        # Chuyển đổi ảnh PIL sang OpenCV
        img_cv = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)

        # Kernel làm nét
        kernel = np.array([[0, -1, 0],
                            [-1, 5,-1],
                            [0, -1, 0]])

        # Áp dụng kernel làm nét
        sharpened = cv2.filter2D(img_cv, -1, kernel)

        # Điều chỉnh độ tương phản bằng cách sử dụng hàm convertTo của OpenCV
        # Tham số alpha là hệ số độ tương phản (1.0 giữ nguyên, >1.0 tăng tương phản)
        alpha = 1.5  # Điều chỉnh độ tương phản (1.0 là không thay đổi)
        contrast_img = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=0)

        # Chuyển lại sang ảnh PIL
        img_rgb = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(img_rgb)

        # Reset zoom và vị trí để hiển thị toàn ảnh làm nét
        self.zoom_slider.set(1)
        self.zoom_factor = 1
        self.offset_x = 0
        self.offset_y = 0
        self.image_position = (0, 0)

        # Hiển thị ảnh sau khi làm nét và tăng cường độ tương phản
        self.display_image()

        messagebox.showinfo("Thông báo", "Đã làm nét và tăng cường độ tương phản ảnh.")


    def reset_image(self):
        if not hasattr(self, "original_image") or self.original_image is None:
            messagebox.showerror("Lỗi", "Chưa có ảnh gốc để khôi phục.")
            return

        # Đặt lại ảnh gốc
        self.image = self.original_image.copy()

        # Reset zoom và vị trí
        self.zoom_slider.set(1)
        self.zoom_factor = 1
        self.offset_x = 0
        self.offset_y = 0
        self.image_position = (0, 0)

        # Hiển thị ảnh gốc
        self.display_image()

        messagebox.showinfo("Thông báo", "Đã đặt lại ảnh gốc.")

    def recognize_license_plate(self):
        if not self.image:
            messagebox.showerror("Lỗi", "Vui lòng chọn hoặc chụp ảnh trước!")
            return

        # Phóng to ảnh gốc theo zoom_factor nếu > 1
        if self.zoom_factor > 1:
            width, height = self.image.size
            new_width = int(width * self.zoom_factor)
            new_height = int(height * self.zoom_factor)
            zoomed_image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            open_cv_image = cv2.cvtColor(np.array(zoomed_image), cv2.COLOR_RGB2BGR)
        else:
            open_cv_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)

        # Gọi hàm nhận diện biển số xe từ LicensePlateRecognizer
        processed_frame, license_plate_texts = self.plate_recognizer.detect_license_plate_and_text(open_cv_image)

        # Hiển thị ảnh đã xử lý
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        processed_image = Image.fromarray(processed_frame_rgb)
        processed_image = processed_image.resize((self.image_label.winfo_width(), self.image_label.winfo_height()), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(image=processed_image)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

        if license_plate_texts:
            messagebox.showinfo("Kết quả nhận diện", f"Biển số xe: {', '.join(license_plate_texts)}")
            cv2.imwrite("detected_license_plate.jpg", processed_frame)
        else:
            messagebox.showinfo("Kết quả nhận diện", "Không nhận diện được biển số xe.")


if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()
