from ultralytics import YOLO
import cv2, os, numpy as np, easyocr, csv
from datetime import datetime
from typing import Optional

model = YOLO("best.pt")
reader = easyocr.Reader(['en'], gpu=False)

def process_video(video_path: str, output_dir: str = "hasil_deteksi_video2") -> str:
    os.makedirs(output_dir, exist_ok=True)
    crop_dir = os.path.join(output_dir, "crop")
    hd_dir = os.path.join(output_dir, "hd")
    thresh_dir = os.path.join(output_dir, "threshold")
    deteksi_dir = os.path.join(output_dir, "deteksi")
    for d in [crop_dir, hd_dir, thresh_dir, deteksi_dir]:
        os.makedirs(d, exist_ok=True)

    csv_path = os.path.join(output_dir, "hasil_video.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Waktu", "Frame", "Label YOLO", "Conf YOLO",
                             "Crop Path", "HD Path", "Threshold Path",
                             "OCR HD", "Conf HD", "OCR Threshold", "Conf Threshold"])

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_interval = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        results = model.predict(frame, conf=0.5, device="cpu")[0]
        if results.boxes:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_yolo = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                crop = frame[y1:y2, x1:x2]
                crop_path = os.path.join(crop_dir, f"crop_{frame_count}_{i+1}.png")
                cv2.imwrite(crop_path, crop)

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, thresh_rot = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                cnts, _ = cv2.findContours(thresh_rot, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                rotated = crop.copy()
                if cnts:
                    all_cnt = np.vstack(cnts)
                    rect = cv2.minAreaRect(all_cnt)
                    angle = rect[-1]
                    if angle < -45:
                        angle = 90 + angle
                    elif angle > 45:
                        angle = angle - 90
                    h, w = crop.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                upscale = cv2.resize(rotated, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                lab = cv2.cvtColor(upscale, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
                hd = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
                hd = cv2.filter2D(hd, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

                hd_path = os.path.join(hd_dir, f"hd_{frame_count}_{i+1}.png")
                cv2.imwrite(hd_path, hd)

                gray_hd = cv2.cvtColor(hd, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray_hd, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 41, 15)
                thresh = thresh[:int(thresh.shape[0] * 0.75), :]
                thresh_path = os.path.join(thresh_dir, f"thresh_{frame_count}_{i+1}.png")
                cv2.imwrite(thresh_path, thresh)

                result_hd = reader.readtext(hd)
                result_t = reader.readtext(thresh)

                texts_hd = [t[1] for t in result_hd if t[2] > 0.5]
                confs_hd = [t[2] for t in result_hd if t[2] > 0.5]
                text_hd = ' '.join(texts_hd)
                avg_hd = sum(confs_hd) / len(confs_hd) if confs_hd else 0

                texts_t = [t[1] for t in result_t if t[2] > 0.5]
                confs_t = [t[2] for t in result_t if t[2] > 0.5]
                text_t = ' '.join(texts_t)
                avg_t = sum(confs_t) / len(confs_t) if confs_t else 0

                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        frame_count, label, f"{conf_yolo:.2f}",
                        crop_path, hd_path, thresh_path,
                        text_hd, f"{avg_hd:.2f}", text_t, f"{avg_t:.2f}"
                    ])

                deteksi_path = os.path.join(deteksi_dir, f"deteksi_{frame_count}.jpg")
                cv2.imwrite(deteksi_path, frame)

    cap.release()
    return csv_path
