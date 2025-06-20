from ultralytics import YOLO

# Load model (gunakan model dasar YOLOv11, atau custom yang kamu simpan)
model = YOLO('yolo11n.pt')  # atau 'yolo11s.pt', dsb.

# Mulai training
model.train(
    data='D:\PA-D4-LJ\Plat-Nomor\PlatNomorPA.v2i.yolov8\data.yaml', # path ke file YAML Roboflow
    epochs=50,
    imgsz=640,
    batch=16,
)