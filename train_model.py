import os
from ultralytics import YOLO


def main():
    model = YOLO('yolov8n.pt')
    
    model.train(
        data='D:\\DATN_nop_bai\\Dataset\\data.yaml',
        epochs=40,
        cache=False,
        imgsz=640,
        batch=2,           # điều chỉnh nếu RAM bị đầy
        name='lp_yolov8n',   # tên thư mục lưu kết quả
        device='0',
        amp=False,
        workers=4,
        # resume=True
    )

if __name__ == '__main__':
    main()