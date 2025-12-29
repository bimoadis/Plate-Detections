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
VIDEO_FOLDER = os.path.join(UPLOAD_FOLDER, "uploaded_videos")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

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
    if file.filename == '':
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400
    
    # Generate filename dengan timestamp untuk menghindari duplikasi
    from datetime import datetime
    original_filename = secure_filename(file.filename)
    name, ext = os.path.splitext(original_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_filename = f"{name}_{timestamp}{ext}"
    
    # Simpan video di folder uploaded_videos
    video_path = os.path.join(VIDEO_FOLDER, saved_filename)
    file.save(video_path)
    
    # Simpan path video untuk deteksi nanti
    video_path_file = os.path.join(UPLOAD_FOLDER, "current_video.txt")
    with open(video_path_file, 'w') as f:
        f.write(video_path)
    
    # Simpan info video untuk akses nanti
    video_info = {
        "filename": saved_filename,
        "original_filename": original_filename,
        "path": video_path,
        "uploaded_at": datetime.now().isoformat()
    }
    import json
    video_info_file = os.path.join(UPLOAD_FOLDER, "current_video_info.json")
    with open(video_info_file, 'w') as f:
        json.dump(video_info, f)
    
    return jsonify({
        "message": "✅ Video berhasil diupload dan disimpan",
        "filename": saved_filename,
        "original_filename": original_filename
    })

def run_detection(video_path, output_dir):
    """Jalankan deteksi di background thread"""
    global detection_status
    try:
        detection_status["is_processing"] = True
        detection_status["error"] = None
        detection_status["message"] = ""
        
        print(f"🚀 Memulai deteksi untuk: {video_path}")
        hasil_csv = process_video(video_path, output_dir=output_dir, gpu_id=0)
        
        detection_status["is_processing"] = False
        detection_status["message"] = ""
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

@app.route('/get-current-video', methods=['GET'])
def get_current_video():
    """Mendapatkan informasi video yang sedang aktif"""
    import json
    video_info_file = os.path.join(UPLOAD_FOLDER, "current_video_info.json")
    video_path_file = os.path.join(UPLOAD_FOLDER, "current_video.txt")
    
    # Cek apakah ada video info
    if os.path.exists(video_info_file):
        try:
            with open(video_info_file, 'r') as f:
                video_info = json.load(f)
            
            # Cek apakah file video masih ada
            if os.path.exists(video_info.get("path", "")):
                # Generate URL untuk video
                folder_name = os.path.basename(UPLOAD_FOLDER)
                video_url = f"/{folder_name}/uploaded_videos/{video_info['filename']}"
                return jsonify({
                    "exists": True,
                    "video_info": video_info,
                    "video_url": video_url
                })
        except Exception as e:
            print(f"Error reading video info: {e}")
    
    # Fallback: cek current_video.txt
    if os.path.exists(video_path_file):
        try:
            with open(video_path_file, 'r') as f:
                video_path = f.read().strip()
            if os.path.exists(video_path):
                filename = os.path.basename(video_path)
                folder_name = os.path.basename(UPLOAD_FOLDER)
                video_url = f"/{folder_name}/uploaded_videos/{filename}"
                return jsonify({
                    "exists": True,
                    "video_info": {
                        "filename": filename,
                        "path": video_path
                    },
                    "video_url": video_url
                })
        except Exception as e:
            print(f"Error reading video path: {e}")
    
    return jsonify({"exists": False})

@app.route('/<folder_name>/uploaded_videos/<filename>')
def get_uploaded_video(folder_name, filename):
    """Serve uploaded video file"""
    if folder_name != os.path.basename(UPLOAD_FOLDER):
        return jsonify({"error": "Invalid folder"}), 404
    
    video_path = os.path.join(VIDEO_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404
    
    response = send_from_directory(VIDEO_FOLDER, filename)
    # Add CORS headers for video requests
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response


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
        if row.get("Deteksi Path"):
            row["Deteksi Path"] = f"/{folder_name}/deteksi/{os.path.basename(row['Deteksi Path'])}"

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
    
    response = send_from_directory(folder_path, filename)
    # Add CORS headers for image requests
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response


@app.route('/get-metrics', methods=['GET'])
def get_metrics():
    """Mengambil metrik evaluasi OCR dari file metrics_evaluasi.txt"""
    metrics_file = os.path.join(UPLOAD_FOLDER, "metrics_evaluasi.txt")
    
    if not os.path.exists(metrics_file):
        return jsonify({
            "error": "File metrik evaluasi tidak ditemukan",
            "metrics": None
        }), 404
    
    try:
        # Parse file metrics_evaluasi.txt
        metrics = {
            'crop': {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
            'hd': {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
            'sr': {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        }
        
        with open(metrics_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                # Format: "OCR Crop            85.50              82.30              83.88"
                if 'OCR Crop' in line and not line.startswith('Metode'):
                    # Split dan ambil nilai numerik
                    parts = [p for p in line.split() if p.replace('.', '').isdigit()]
                    if len(parts) >= 3:
                        metrics['crop']['precision'] = float(parts[0])
                        metrics['crop']['recall'] = float(parts[1])
                        metrics['crop']['f1_score'] = float(parts[2])
                elif 'OCR HD' in line and not line.startswith('Metode'):
                    parts = [p for p in line.split() if p.replace('.', '').isdigit()]
                    if len(parts) >= 3:
                        metrics['hd']['precision'] = float(parts[0])
                        metrics['hd']['recall'] = float(parts[1])
                        metrics['hd']['f1_score'] = float(parts[2])
                elif 'OCR SR' in line and not line.startswith('Metode'):
                    parts = [p for p in line.split() if p.replace('.', '').isdigit()]
                    if len(parts) >= 3:
                        metrics['sr']['precision'] = float(parts[0])
                        metrics['sr']['recall'] = float(parts[1])
                        metrics['sr']['f1_score'] = float(parts[2])
        
        return jsonify({
            "metrics": metrics,
            "message": "Metrik evaluasi berhasil diambil"
        })
    except Exception as e:
        return jsonify({
            "error": f"Error membaca file metrik: {str(e)}",
            "metrics": None
        }), 500


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
    
    # Hapus file metrics juga
    metrics_file = os.path.join(UPLOAD_FOLDER, 'metrics_evaluasi.txt')
    if os.path.exists(metrics_file):
        os.remove(metrics_file)
        deleted.append('Metrics')
    
    # Hapus current_video.txt tapi JANGAN hapus video yang sudah diupload
    video_path_file = os.path.join(UPLOAD_FOLDER, "current_video.txt")
    if os.path.exists(video_path_file):
        os.remove(video_path_file)
        deleted.append('Current Video Path')

    return jsonify({"message": f"Data dihapus: {', '.join(deleted)}"})

if __name__ == '__main__':
    # Set timeout lebih lama untuk request yang berat
    # Threaded=True untuk handle multiple requests
    app.run(debug=True, port=8000, threaded=True)
