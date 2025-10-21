from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import shutil
from werkzeug.utils import secure_filename
from detect_plate import process_video  # panggil fungsi deteksi
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.getenv("OUTPUT_DIR", "hasil_deteksi_video2")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    temp_path = f"temp_{filename}"
    file.save(temp_path)

    hasil_csv = process_video(temp_path)
    return jsonify({"message": "✅ Proses selesai", "csv_path": hasil_csv})


@app.route('/get-results', methods=['GET'])
def get_results():
    import csv
    csv_path = os.path.join(UPLOAD_FOLDER, "hasil_video.csv")
    if not os.path.exists(csv_path):
        return jsonify({"data": []})

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # konversi path ke URL
    for row in data:
        if row.get("Crop Path"):
            row["Crop Path"] = f"/hasil_deteksi_video/crop/{os.path.basename(row['Crop Path'])}"
        if row.get("HD Path"):
            row["HD Path"] = f"/hasil_deteksi_video/hd/{os.path.basename(row['HD Path'])}"
        if row.get("Threshold Path"):
            row["Threshold Path"] = f"/hasil_deteksi_video/threshold/{os.path.basename(row['Threshold Path'])}"

    return jsonify({"data": data})



@app.route('/get-csv', methods=['GET'])
def get_csv():
    return send_from_directory(UPLOAD_FOLDER, "hasil_video.csv", as_attachment=True)


@app.route('/hasil_deteksi_video/<folder>/<filename>')
def get_image(folder, filename):
    folder_path = os.path.join(UPLOAD_FOLDER, folder)
    return send_from_directory(folder_path, filename)


@app.route('/clear-results', methods=['DELETE'])
def clear_results():
    subfolders = ['crop', 'hd', 'threshold', 'deteksi']
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
    app.run(debug=True, port=8000)
