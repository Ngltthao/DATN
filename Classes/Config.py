import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_DIR = os.path.join(BASE_DIR,"Model")
MODEL_DETECT_LICENSE_PLATE = os.path.join(MODEL_DIR, "runs\\detect\\lp_yolov8n2\\weights\\best.pt")
MODEL_DETECT_CHAR = os.path.join(MODEL_DIR, "training_data.json")

DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
TRAINING_DATA_DIR = os.path.join(DATASET_DIR, "train")
TESTING_DATA_DIR = os.path.join(DATASET_DIR, "test")

WINDOW_TITLE = "NHẬN DIỆN BIỂN SỐ XE"
WINDOW_SIZE = "1000X600"
BACKGROUND_COLOR = "#ECF0F1"

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)  # Kích cỡ bộ lọc Gauss
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

NAME_CAPTURE_BUTTON = "Chụp Hình"
NAME_RECORD_BUTTON = "Quay Video"
NAME_UPLOAD_BUTTON = "Tải Lên"
NAME_START_CAMERA_BUTTON = "Bắt đầu Camera"
NAME_STOP_CAMERA_BUTTON = "Tắt Camera"
NAME_ZOOM_SLIDER = "Phóng To"
NAME_SHARPEN_BUTTON = "Làm nét ảnh"
NAME_RESET_BUTTON = "Đặt lại ảnh"
NAME_RECOGNIZE_BUTTON = "Nhận diện Biển Số"