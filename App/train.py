import numpy as np
import cv2

# Đọc dữ liệu huấn luyện
flattened_images = np.loadtxt('flattened_images.txt', dtype=np.float32)  # Chuyển dữ liệu thành float32
classifications = np.loadtxt('classifications.txt', dtype=np.float32)  # Chuyển dữ liệu thành float32

# Đảm bảo rằng classifications có kiểu dữ liệu đúng (CV_32S)
classifications = classifications.reshape((classifications.size, 1))

# Huấn luyện mô hình (ví dụ: KNN)
knn = cv2.ml.KNearest_create()
knn.train(flattened_images, cv2.ml.ROW_SAMPLE, classifications)

# Lưu mô hình đã huấn luyện
knn.save('modeltrain.xml')

