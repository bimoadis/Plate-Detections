from ultralytics import YOLO
import cv2, os, numpy as np, csv
from datetime import datetime
import torch

# Fix torchvision compatibility issue before importing basicsr
def fix_basicsr_torchvision():
    """Auto-fix basicsr compatibility with newer torchvision versions"""
    try:
        import basicsr
        basicsr_path = os.path.dirname(basicsr.__file__)
        degradations_path = os.path.join(basicsr_path, 'data', 'degradations.py')
        
        if os.path.exists(degradations_path):
            with open(degradations_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if fix is needed
            if 'from torchvision.transforms.functional_tensor import' in content:
                content = content.replace(
                    'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
                    'from torchvision.transforms.functional import rgb_to_grayscale'
                )
                content = content.replace(
                    'functional_tensor.rgb_to_grayscale',
                    'rgb_to_grayscale'
                )
                
                with open(degradations_path, 'w', encoding='utf-8') as f:
                    f.write(content)
    except Exception:
        pass  # Silently fail if fix cannot be applied

# Apply fix before importing
fix_basicsr_torchvision()

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# =========================================
# 🔹 Cek dan Setup CUDA Device
# =========================================
def get_device(gpu_id=0):
    """Mendapatkan device yang tepat (CUDA jika tersedia, CPU jika tidak)"""
    if torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            print(f"⚠️  PERINGATAN: GPU ID {gpu_id} tidak tersedia, menggunakan GPU 0")
            gpu_id = 0
        device_name = torch.cuda.get_device_name(gpu_id)
        print(f"🎮 Menggunakan GPU {gpu_id} untuk YOLO: {device_name}")
        return gpu_id  # YOLO menggunakan integer untuk GPU ID
    else:
        print("⚠️  PERINGATAN: CUDA tidak tersedia, menggunakan CPU (lambat!)")
        return "cpu"

# Inisialisasi device
_device = get_device(0)

# =========================================
# 🔹 Load Model YOLO
# =========================================
model_plate = YOLO("runs11s/detect/train/weights/best.pt")   # deteksi plat nomor
model_ocr = YOLO("OCRCUSTOMV5/content/runs/detect/train2/weights/best.pt")
  # deteksi karakter OCR

# =========================================
# 🔹 Load Model SR (Real-ESRGAN)
# =========================================
_sr_upsampler = None
def get_sr_upsampler(gpu_id=0):
    """Initialize dan return SR upsampler (singleton pattern)"""
    global _sr_upsampler
    if _sr_upsampler is None:
        # Cek CUDA
        use_cuda = torch.cuda.is_available()
        
        if use_cuda:
            # Validasi GPU ID
            if gpu_id >= torch.cuda.device_count():
                print(f"⚠️  PERINGATAN: GPU ID {gpu_id} tidak tersedia, menggunakan GPU 0")
                gpu_id = 0
            device_name = torch.cuda.get_device_name(gpu_id)
            print(f"🎮 Menggunakan GPU {gpu_id} untuk SR: {device_name}")
        else:
            print("⚠️  PERINGATAN: CUDA tidak tersedia untuk SR, menggunakan CPU (lambat!)")
            gpu_id = None
        
        # Path model SR
        model_path = os.path.join('weights', 'net_g_latest.pth')
        if not os.path.isfile(model_path):
            print(f"❌ ERROR: Model SR tidak ditemukan di: {model_path}")
            return None
        
        # Buat model RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        
        # Buat upsampler
        _sr_upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True if use_cuda else False,  # FP16 hanya untuk GPU
            gpu_id=gpu_id
        )
        print("✅ Model SR berhasil dimuat!")
    return _sr_upsampler

# =========================================
# 🔹 Fungsi peningkatan kualitas (HD)
# =========================================
def enhance_hd(img):
    """Meningkatkan kualitas gambar agar lebih jelas untuk OCR."""
    # Resize ke ukuran lebih besar
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    # Sharpening (penajaman)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    # Denoise ringan
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return img

# =========================================
# 🔹 Fungsi baca OCR YOLO
# =========================================
def run_custom_ocr(image):
    """Mendeteksi karakter dari gambar menggunakan YOLO OCR custom."""
    results_ocr = model_ocr.predict(source=image, conf=0.3, device=_device)[0]

    boxes = results_ocr.boxes.xyxy.cpu().numpy()
    classes = results_ocr.boxes.cls.cpu().numpy()
    names = results_ocr.names

    detections = []
    for box2, cls2 in zip(boxes, classes):
        x1c, y1c, x2c, y2c = box2
        detections.append((x1c, names[int(cls2)]))

    # Urutkan karakter dari kiri ke kanan
    detections = sorted(detections, key=lambda x: x[0])
    text = "".join([ch for _, ch in detections])
    return text.strip()

# =========================================
# 🔹 Fungsi LR to SR
# =========================================
def enhance_sr(image, gpu_id=0):
    """Meningkatkan kualitas gambar menggunakan Real-ESRGAN (Super Resolution)."""
    upsampler = get_sr_upsampler(gpu_id)
    if upsampler is None:
        return None
    
    try:
        # Proses super resolution
        sr_image, _ = upsampler.enhance(image, outscale=4)
        return sr_image
    except Exception as e:
        print(f"⚠️  Error saat SR: {e}")
        return None

# =========================================
# 🔹 Fungsi utama
# =========================================
def process_video(video_path: str, output_dir: str = "hasil_deteksi_video", gpu_id: int = 0) -> str:
    """Memproses video untuk deteksi plat nomor dengan CUDA jika tersedia"""
    global _device
    
    # Update device jika gpu_id berbeda
    if isinstance(_device, int) and _device != gpu_id:
        _device = get_device(gpu_id)
    elif not isinstance(_device, int) and torch.cuda.is_available():
        _device = get_device(gpu_id)
    
    print(f"📹 Memproses video: {video_path}")
    print(f"🔧 Device yang digunakan: {_device}")
    
    os.makedirs(output_dir, exist_ok=True)
    crop_dir = os.path.join(output_dir, "crop")
    hd_dir = os.path.join(output_dir, "hd")
    sr_dir = os.path.join(output_dir, "sr")
    deteksi_dir = os.path.join(output_dir, "deteksi")

    for d in [crop_dir, hd_dir, sr_dir, deteksi_dir]:
        os.makedirs(d, exist_ok=True)

    csv_path = os.path.join(output_dir, "hasil_video.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Waktu", "Frame", "Label Plat", "Conf YOLO",
                "OCR Crop", "OCR HD", "OCR SR",
                "Crop Path", "HD Path", "SR Path", "Deteksi Path"
            ])

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_interval = 25  # proses tiap 25 frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        # =========================================
        # 1️⃣ DETEKSI PLAT NOMOR
        # =========================================
        results_plate = model_plate.predict(frame, conf=0.5, device=_device)[0]
        if not results_plate.boxes:
            continue

        for i, box in enumerate(results_plate.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_yolo = float(box.conf[0])
            cls = int(box.cls[0])
            label = model_plate.names[cls]

            # =========================================
            # 2️⃣ Simpan crop
            # =========================================
            crop = frame[y1:y2, x1:x2]
            crop_path = os.path.join(crop_dir, f"crop_{frame_count}_{i+1}.png")
            cv2.imwrite(crop_path, crop)

            # =========================================
            # 3️⃣ Buat versi HD dan simpan
            # =========================================
            hd_crop = enhance_hd(crop)
            hd_path = os.path.join(hd_dir, f"hd_{frame_count}_{i+1}.png")
            cv2.imwrite(hd_path, hd_crop)

            # =========================================
            # 4️⃣ Jalankan OCR pada crop & HD crop
            # =========================================
            plate_text_crop = run_custom_ocr(crop)
            plate_text_hd = run_custom_ocr(hd_crop)

            # =========================================
            # 5️⃣ Buat versi SR dari HD dan simpan
            # =========================================
            sr_crop = enhance_sr(hd_crop, gpu_id=gpu_id)
            sr_path = None
            plate_text_sr = ""
            
            if sr_crop is not None:
                sr_path = os.path.join(sr_dir, f"sr_{frame_count}_{i+1}.png")
                cv2.imwrite(sr_path, sr_crop)
                
                # =========================================
                # 6️⃣ Jalankan OCR pada SR crop
                # =========================================
                plate_text_sr = run_custom_ocr(sr_crop)

            # =========================================
            # 7️⃣ Simpan frame hasil deteksi
            # =========================================
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Prioritas: SR > HD > Crop
            text_disp = plate_text_sr if plate_text_sr else (plate_text_hd if plate_text_hd else plate_text_crop)
            cv2.putText(frame, text_disp, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            deteksi_path = os.path.join(deteksi_dir, f"deteksi_{frame_count}_{i+1}.jpg")
            cv2.imwrite(deteksi_path, frame)

            # =========================================
            # 8️⃣ Simpan hasil ke CSV
            # =========================================
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    frame_count, label, f"{conf_yolo:.2f}",
                    plate_text_crop, plate_text_hd, plate_text_sr,
                    crop_path, hd_path, sr_path if sr_path else "", deteksi_path
                ])

            print(f"[Frame {frame_count}] Plat: {text_disp} | Conf: {conf_yolo:.2f} | OCR: Crop={plate_text_crop}, HD={plate_text_hd}, SR={plate_text_sr}")

    cap.release()
    print("\n✅ Selesai! Hasil disimpan di:", csv_path)
    return csv_path
