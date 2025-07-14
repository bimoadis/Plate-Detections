from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
from detect_plate import process_video
from fastapi.responses import FileResponse

app = FastAPI()

# Bolehkan akses dari frontend Nuxt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti dengan URL Nuxt jika produksi
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    hasil_csv = process_video(temp_path)
    return {"message": "✅ Proses selesai", "csv_path": hasil_csv}


@app.get("/get-results")
def get_csv():
    path = "hasil_deteksi_video2/hasil_video.csv"
    if not os.path.exists(path):
        return {"message": "Belum ada hasil."}
    with open(path, "r") as file:
        lines = file.readlines()
        header = lines[0].strip().split(",")
        data = [dict(zip(header, row.strip().split(","))) for row in lines[1:]]
    return {"data": data}


@app.get("/images/{folder}/{filename}")
def get_image(folder: str, filename: str):
    path = f"hasil_deteksi_video2/{folder}/{filename}"
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "File tidak ditemukan."}