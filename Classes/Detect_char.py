import cv2, json
import easyocr
import numpy as np
import re
import sys
from ultralytics import YOLO  # Thêm dòng này ở đầu file
import Classes.Config as Config

# Preprocess functions
def preprocess(imgOriginal):
    GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)   # Kích cỡ bộ lọc Gauss
    ADAPTIVE_THRESH_BLOCK_SIZE = 19
    ADAPTIVE_THRESH_WEIGHT = 9

    def extractValue(imgOriginal):
        height, width, numChannels = imgOriginal.shape
        imgHSV = np.zeros((height, width, 3), np.uint8)
        imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
        imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
        return imgValue

    def maximizeContrast(imgGrayscale):
        height, width = imgGrayscale.shape
        imgTopHat = np.zeros((height, width, 1), np.uint8)
        imgBlackHat = np.zeros((height, width, 1), np.uint8)
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations=10)
        imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations=10)
        imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
        imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
        return imgGrayscalePlusTopHatMinusBlackHat

    imgGrayscale = extractValue(imgOriginal)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    height, width = imgGrayscale.shape
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh

# LicensePlateRecognizer class

class LicensePlateRecognizer:
    def __init__(self, languages=['en', 'vi']):
        self.reader = easyocr.Reader(languages)  # Khởi tạo EasyOCR cho nhận diện văn bản
        # self.model = YOLO("D:\\DATN_nop_bai\\Model\\runs\\detect\\lp_yolov8n2\\weights\\best.pt")  # Mô hình YOLO
        self.model = YOLO(Config.MODEL_DETECT_LICENSE_PLATE)
        print("Mô hình YOLO đã được tải thành công.")
        
        # Load dữ liệu huấn luyện từ JSON
        self.load_training_data("C:\\DATN_nop_bai\\Model\\training_data.json")

    def load_training_data(self, json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

                images = [item["image"] for item in data]
                labels = [ord(item["label"]) for item in data]  # Đổi label thành mã ASCII

                self.flattened_images = np.array(images, np.float32)
                self.classifications = np.array(labels, np.float32).reshape(-1, 1)

                print("✅ Đã tải dữ liệu huấn luyện từ JSON.")
                print("📦 Số lượng mẫu huấn luyện:", len(labels))
        except Exception as e:
            print("❌ Lỗi khi tải dữ liệu JSON:", e)
            self.classifications = None
            self.flattened_images = None

    def preprocess_image(self, img):
        imgGrayscale, imgThresh = preprocess(img)
        return imgThresh

    # Lọc và chỉ giữ lại ký tự hợp lệ như 68H-125.23
    def filter_text(self, text):
        text = text.upper()  # Chuyển toàn bộ văn bản thành chữ hoa
        # Loại bỏ các ký tự không phải chữ cái, số, dấu gạch ngang hoặc dấu chấm
        text = re.sub(r'[^A-Z0-9\-\.]', '', text)
        
        # Kiểm tra xem có dấu chấm hay không
        if '.' in text:
            # Nếu có dấu chấm, đảm bảo chỉ có một dấu chấm
            text = re.sub(r'\.(?=.*\.)', '', text)
        return text

    def detect_license_plate_and_text(self, frame):
        if frame is None:
            print("Ảnh không hợp lệ!")
            return frame, []

        # Nhận diện biển số với mô hình YOLO
        results = self.model(frame, conf=0.5, verbose=False)

        detected_texts = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                plate_img = frame[y1:y2, x1:x2]  # Cắt biển số từ ảnh

                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Đọc văn bản từ ảnh biển số
                result_text = self.reader.readtext(plate_img)
                box_texts = []
                for detection in result_text:
                    text = self.filter_text(detection[1])  # Lọc văn bản
                    if text:  # Nếu có văn bản hợp lệ
                        print("Biển số xe nhận diện được:", text)
                        box_texts.append(text)
                if box_texts:
                    # Vẽ hình chữ nhật quanh biển số và hiển thị văn bản nhận diện
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, ' - '.join(box_texts), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    detected_texts.extend(box_texts) 

        # Nếu không có biển số nào được nhận diện, thông báo cho người dùng
        if not detected_texts:
            print("Không nhận diện được biển số xe.")

        return frame, detected_texts

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print("Không thể đọc ảnh!")
            return None

        processed_img, detected_texts = self.detect_license_plate_and_text(img)
        output_path = 'output_license_plate.jpg'
        cv2.imwrite(output_path, processed_img)
        print(f"Đã lưu ảnh đã nhận diện vào file: {output_path}")

        return detected_texts

    def show_live_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở camera!")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ camera!")
                break

            processed_frame, detected_texts = self.detect_license_plate_and_text(frame)
            cv2.imshow("Live License Plate Detection", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.imwrite("output_live_license_plate.jpg", processed_frame)
                print("Đã lưu ảnh đã nhận diện vào file: output_live_license_plate.jpg")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()