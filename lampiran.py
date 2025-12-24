import csv
import os
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage

# =========================
# PATH FILE
# =========================
csv_path = r"D:\PA-D4-LJ\Plat-Nomor\hasil_deteksi_video\hasil_video_wardhni siang.csv"
excel_path = r"D:\PA-D4-LJ\Plat-Nomor\hasil_deteksi_video\hasil_video.xlsx"

# =========================
# BUAT EXCEL
# =========================
wb = Workbook()
ws = wb.active
ws.title = "Hasil Deteksi Plat"

headers = [
    "Waktu", "Frame", "Label Plat", "Conf YOLO",
    "OCR Crop", "OCR HD", "OCR SR",
    "Crop Image", "HD Image", "SR Image"
]
ws.append(headers)

# =========================
# BACA CSV
# =========================
with open(csv_path, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    excel_row = 2

    for row in reader:
        # Tulis data teks
        ws.append([
            row["Waktu"],
            row["Frame"],
            row["Label Plat"],
            float(row["Conf YOLO"]) if row["Conf YOLO"] else "",
            row["OCR Crop"],
            row["OCR HD"],
            row["OCR SR"]
        ])

        # =========================
        # MASUKKAN GAMBAR
        # =========================
        image_columns = {
            "Crop Path": "H",
            "HD Path": "I",
            "SR Path": "J"
        }

        for csv_col, excel_col in image_columns.items():
            img_path = row[csv_col]

            if img_path and os.path.exists(img_path):
                img = XLImage(img_path)
                img.width = 120
                img.height = 60
                ws.add_image(img, f"{excel_col}{excel_row}")

        ws.row_dimensions[excel_row].height = 55
        excel_row += 1

# =========================
# SIMPAN FILE
# =========================
wb.save(excel_path)
print("✅ Convert CSV ke Excel + gambar berhasil!")
print("📁 Output:", excel_path)
