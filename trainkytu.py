import cv2
import os
import numpy as np

# Đọc ảnh chứa nhiều ký tự
img_path = r'C:\DATN\App\training_chars.png'  # Đảm bảo sử dụng đường dẫn đúng
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Tiền xử lý ảnh: làm sạch và chuyển về ảnh nhị phân
_, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

# Sử dụng Canny để phát hiện biên
edges = cv2.Canny(img_binary, 100, 200)

# Tìm các contour (các vùng chứa ký tự)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tạo thư mục để lưu ảnh con
output_folder = r'C:\DATN\App\kytu'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Duyệt qua các contour để cắt các ký tự
for idx, contour in enumerate(contours):
    # Vẽ hình chữ nhật bao quanh từng ký tự
    x, y, w, h = cv2.boundingRect(contour)
    if w > 5 and h > 5:  # Kiểm tra kích thước của bounding box
        # Cắt vùng chứa ký tự
        char_img = img[y:y+h, x:x+w]

        # Resize ảnh về kích thước chuẩn (28x28 pixel)
        char_img_resized = cv2.resize(char_img, (28, 28))

        # Lưu ảnh ký tự vào thư mục
        char_img_filename = os.path.join(output_folder, f'char_{idx}.png')
        cv2.imwrite(char_img_filename, char_img_resized)

print("Đã tách các ký tự và lưu vào thư mục C:\\DATN\\App\\kytu!")

import json

# Danh sách các ký tự theo thứ tự (có thể thay đổi tuỳ theo nhu cầu)
valid_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Gán nhãn cho các ký tự và lưu trữ dữ liệu huấn luyện
training_data = []

for idx, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    if w > 5 and h > 5:  # Kiểm tra kích thước của bounding box
        # Lấy ký tự từ danh sách valid_chars
        char_label = valid_chars[idx % len(valid_chars)]

        # Cắt và resize ảnh ký tự
        char_img = img[y:y+h, x:x+w]
        char_img_resized = cv2.resize(char_img, (28, 28))

        # Chuẩn hóa ảnh (0-255 -> 0-1)
        char_img_normalized = char_img_resized / 255.0

        # Flatten ảnh thành vector
        char_img_vector = char_img_normalized.flatten()

        # Thêm vào dữ liệu huấn luyện
        training_data.append({'image': char_img_vector.tolist(), 'label': char_label})

# Lưu dữ liệu huấn luyện vào file JSON
with open(r'C:\DATN\App\kytu\training_data.json', 'w') as f:
    json.dump(training_data, f)

print("Tạo file huấn luyện thành công và lưu vào C:\\DATN\\App\\kytu\\training_data.json!")
