from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, shutil

# Import fungsi proses
from detect_plate import process_video

app = FastAPI()

# === CORS (Frontend Vue) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # ubah sesuai alamat frontend kamu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_videos"
OUTPUT_DIR = "hasil_deteksi_video"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. Upload Video ===
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename
    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": "✅ Upload berhasil", "filename": filename}


# === 2. Proses Video (memanggil process_video dari detect_plate.py) ===
@app.post("/process-video")
async def process_video_endpoint(filename: str = Form(...)):
    input_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="❌ Video tidak ditemukan.")

    csv_path = process_video(input_path, output_dir=OUTPUT_DIR)
    return {
        "message": "✅ Proses selesai",
        "csv_path": csv_path
    }


# === 3. Ambil CSV hasil deteksi ===
@app.get("/get-csv")
def get_csv():
    csv_path = os.path.join(OUTPUT_DIR, "hasil_video.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV tidak ditemukan")
    return FileResponse(csv_path, filename="hasil_video.csv", media_type="text/csv")


# === 4. Clear hasil deteksi (semua file & gambar) ===
@app.delete("/clear-results")
def clear_results():
    deleted = []
    csv_path = os.path.join(OUTPUT_DIR, "hasil_video.csv")

    if os.path.exists(csv_path):
        os.remove(csv_path)
        deleted.append("CSV")

    for sub in ["crop", "hd", "threshold", "deteksi"]:
        folder = os.path.join(OUTPUT_DIR, sub)
        if os.path.exists(folder):
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))
            deleted.append(sub)

    return {"message": f"Hasil terhapus: {', '.join(deleted)}"}
