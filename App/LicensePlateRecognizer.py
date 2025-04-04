import cv2
import easyocr
import numpy as np
import re
import sys

# Preprocess functions
def preprocess(imgOriginal):
    GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)  # Kích cỡ bộ lọc Gauss
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
        self.reader = easyocr.Reader(languages)
        self.plate_cascade = cv2.CascadeClassifier("C:/DATN/App/modeltrain.xml")
        if self.plate_cascade.empty():
            print("Lỗi: Không thể tải file CascadeClassifier.")
        else:
            print("Mô hình CascadeClassifier đã được tải thành công.")


    def preprocess_image(self, img):
        imgGrayscale, imgThresh = preprocess(img)
        return imgThresh

    def filter_text(self, text):
        text = text.upper()
        text = re.sub(r'[^A-Z0-9]', '', text)  # Loại bỏ ký tự không hợp lệ
        return text

    def detect_license_plate_and_text(self, frame):
        if frame is None:
            print("Ảnh không hợp lệ!")
            return frame, []

        # Xử lý ảnh trước khi nhận diện biển số
        gray = self.preprocess_image(frame)

        # Phát hiện các biển số xe
        plates = self.plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(80, 80))

        detected_texts = []
        for (x, y, w, h) in plates:
            plate_img = frame[y:y + h, x:x + w]  # Cắt ảnh biển số
            result = self.reader.readtext(plate_img)  # Nhận diện văn bản

            for detection in result:
                text = self.filter_text(detection[1])  # Lọc văn bản
                if text:
                    print("Biển số xe nhận diện được:", text)
                    detected_texts.append(text)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Vẽ hình chữ nhật bao quanh biển số
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Hiển thị văn bản biển số

        return frame, detected_texts  # Trả về ảnh đã xử lý và danh sách biển số nhận diện được

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

    def generate_training_data(self):
        print("Bắt đầu tạo dữ liệu huấn luyện...")
        self.generate_data()  # Gọi hàm để tạo dữ liệu huấn luyện
        print("Hoàn tất việc tạo dữ liệu huấn luyện!")

    def generate_data(self):
        MIN_CONTOUR_AREA = 40
        RESIZED_IMAGE_WIDTH = 20
        RESIZED_IMAGE_HEIGHT = 30
        imgTrainingNumbers = cv2.imread("training_chars.png")
        
        imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
        imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)
        imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        imgThreshCopy = imgThresh.copy()

        npaContours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        intClassifications = []
        intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                         ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                         ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                         ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

        for npaContour in npaContours:
            if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
                [intX, intY, intW, intH] = cv2.boundingRect(npaContour)
                cv2.rectangle(imgTrainingNumbers, (intX, intY), (intX + intW, intY + intH), (0, 0, 255), 2)

                imgROI = imgThresh[intY:intY + intH, intX:intX + intW]
                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

                cv2.imshow("imgROI", imgROI)
                cv2.imshow("imgROIResized", imgROIResized)
                cv2.imshow("training_numbers.png", imgTrainingNumbers)

                intChar = cv2.waitKey(0)

                if intChar == 27:
                    sys.exit()
                elif intChar in intValidChars:
                    intClassifications.append(intChar)
                    npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                    npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

        fltClassifications = np.array(intClassifications, np.float32)
        npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

        np.savetxt("classifications.txt", npaClassifications)
        np.savetxt("flattened_images.txt", npaFlattenedImages)

        cv2.destroyAllWindows()

# Kiểm tra kết quả với một ảnh mẫu
if __name__ == "__main__":
    recognizer = LicensePlateRecognizer()

    # Nhận diện biển số từ ảnh
    detected_texts = recognizer.process_image("input_license_plate_image.jpg")
    print("Biển số nhận diện:", detected_texts)

    # Nhận diện trực tiếp từ camera
    # recognizer.show_live_detection()

    # Tạo dữ liệu huấn luyện
    # recognizer.generate_training_data()
