from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import shutil
import threading
import traceback
from werkzeug.utils import secure_filename
from detect_plate import process_video  # panggil fungsi deteksi
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.getenv("OUTPUT_DIR", "hasil_deteksi_video2")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Status tracking untuk deteksi
detection_status = {
    "is_processing": False,
    "error": None,
    "message": None
}

@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{filename}")
    file.save(temp_path)
    
    # Simpan path video untuk deteksi nanti
    video_path_file = os.path.join(UPLOAD_FOLDER, "current_video.txt")
    with open(video_path_file, 'w') as f:
        f.write(temp_path)
    
    return jsonify({"message": "✅ Video berhasil diupload", "filename": filename})

def run_detection(video_path, output_dir):
    """Jalankan deteksi di background thread"""
    global detection_status
    try:
        detection_status["is_processing"] = True
        detection_status["error"] = None
        detection_status["message"] = "Memproses deteksi plat nomor..."
        
        print(f"🚀 Memulai deteksi untuk: {video_path}")
        hasil_csv = process_video(video_path, output_dir=output_dir, gpu_id=0)
        
        detection_status["is_processing"] = False
        detection_status["message"] = "✅ Deteksi plat nomor selesai"
        detection_status["error"] = None
        print(f"✅ Deteksi selesai: {hasil_csv}")
        
    except Exception as e:
        error_msg = f"Error saat deteksi: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        detection_status["is_processing"] = False
        detection_status["error"] = error_msg
        detection_status["message"] = None

@app.route('/detect-plate', methods=['POST'])
def detect_plate():
    global detection_status
    
    # Cek apakah sudah ada proses yang berjalan
    if detection_status["is_processing"]:
        return jsonify({
            "error": "Deteksi sedang berjalan, tunggu hingga selesai",
            "is_processing": True
        }), 409
    
    video_path_file = os.path.join(UPLOAD_FOLDER, "current_video.txt")
    if not os.path.exists(video_path_file):
        return jsonify({"error": "Tidak ada video yang diupload"}), 400
    
    with open(video_path_file, 'r') as f:
        video_path = f.read().strip()
    
    if not os.path.exists(video_path):
        return jsonify({"error": "File video tidak ditemukan"}), 404
    
    # Reset status
    detection_status = {
        "is_processing": True,
        "error": None,
        "message": "Memulai deteksi..."
    }
    
    # Jalankan deteksi di background thread
    thread = threading.Thread(
        target=run_detection,
        args=(video_path, UPLOAD_FOLDER),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        "message": "✅ Deteksi plat nomor dimulai",
        "is_processing": True
    })

@app.route('/detect-status', methods=['GET'])
def detect_status():
    """Cek status deteksi"""
    global detection_status
    return jsonify(detection_status)


@app.route('/get-results', methods=['GET'])
def get_results():
    import csv
    csv_path = os.path.join(UPLOAD_FOLDER, "hasil_video.csv")
    if not os.path.exists(csv_path):
        return jsonify({"data": []})

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # konversi path ke URL (gunakan folder name dari UPLOAD_FOLDER)
    folder_name = os.path.basename(UPLOAD_FOLDER)
    for row in data:
        if row.get("Crop Path"):
            row["Crop Path"] = f"/{folder_name}/crop/{os.path.basename(row['Crop Path'])}"
        if row.get("HD Path"):
            row["HD Path"] = f"/{folder_name}/hd/{os.path.basename(row['HD Path'])}"
        if row.get("SR Path"):
            row["SR Path"] = f"/{folder_name}/sr/{os.path.basename(row['SR Path'])}"
        if row.get("Threshold Path"):
            row["Threshold Path"] = f"/{folder_name}/threshold/{os.path.basename(row['Threshold Path'])}"

    return jsonify({"data": data})



@app.route('/get-csv', methods=['GET'])
def get_csv():
    return send_from_directory(UPLOAD_FOLDER, "hasil_video.csv", as_attachment=True)


@app.route('/<folder_name>/<folder>/<filename>')
def get_image(folder_name, folder, filename):
    # Validasi folder_name sesuai dengan UPLOAD_FOLDER
    if folder_name != os.path.basename(UPLOAD_FOLDER):
        return jsonify({"error": "Invalid folder"}), 404
    
    folder_path = os.path.join(UPLOAD_FOLDER, folder)
    if not os.path.exists(folder_path):
        return jsonify({"error": "Folder not found"}), 404
    return send_from_directory(folder_path, filename)


@app.route('/clear-results', methods=['DELETE'])
def clear_results():
    subfolders = ['crop', 'hd', 'sr', 'threshold', 'deteksi']
    deleted = []

    for folder in subfolders:
        dir_path = os.path.join(UPLOAD_FOLDER, folder)
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, file))
            deleted.append(folder)

    csv_path = os.path.join(UPLOAD_FOLDER, 'hasil_video.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
        deleted.append('CSV')

    return jsonify({"message": f"Data dihapus: {', '.join(deleted)}"})

if __name__ == '__main__':
    # Set timeout lebih lama untuk request yang berat
    # Threaded=True untuk handle multiple requests
    app.run(debug=True, port=8000, threaded=True)
