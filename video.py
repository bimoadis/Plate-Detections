import cv2
import os
from tkinter import Tk, filedialog

# === 1. PILIH VIDEO ===
Tk().withdraw()  # Sembunyikan jendela utama Tkinter
video_path = filedialog.askopenfilename(
    title="Pilih Video",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)

if not video_path:
    print("❌ Tidak ada file yang dipilih.")
    exit()

# === 2. SETUP OUTPUT FOLDER ===
output_folder = "frames_per_30frames_hd_png"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# === 3. BUKA VIDEO ===
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Gagal membuka video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = 30  # Setiap 30 frame

# Ambil resolusi asli video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📹 Resolusi video asli: {width} x {height} | FPS: {fps:.2f}")

frame_count = 0
saved_count = 0

# === 4. SIMPAN FRAME DENGAN RESIZE KE HD (1280x720) ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Resize ke 1280x720 (HD)
        resized_frame = cv2.resize(frame, (1920, 1080))

        filename = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
        cv2.imwrite(filename, resized_frame)
        print(f"✅ Simpan: {filename}")
        saved_count += 1

    frame_count += 1

cap.release()
print("🎉 Selesai menyimpan frame PNG dengan resolusi HD setiap 30 frame.")
