import os
import csv

# File input dan output
input_filename = 'D:\PA-D4-LJ\Plat-Nomor\hasil_deteksi_video\hasil_video.csv'
output_filename = 'hasil.csv'

# # Header tetap 11 kolom
# correct_header = [
#     "Tanggal", "Nama File", "Hasil YOLO", "Confidence YOLO", 
#     "Path Crop", "Path HD", "Path Threshold", "OCR dari HD", 
#     "Conf HD", "OCR dari Threshold", "Conf Threshold"
# ]

correct_header = [
    'Waktu','Frame','Label YOLO','Conf YOLO','Crop Path','HD Path','Threshold Path','OCR HD','Conf HD','OCR Threshold','Conf Threshold'
]

RECORD_SIZE = len(correct_header)
all_final_records = []

if not os.path.exists(input_filename):
    print(f"❌ File '{input_filename}' tidak ditemukan.")
else:
    print(f"📥 Membaca file '{input_filename}'...")

    with open(input_filename, 'r', encoding='utf-8') as file_input:
        next(file_input, None)  # Skip header lama

        for line in file_input:
            line = line.strip()
            if not line:
                continue

            line_for_detection = line.rstrip(';').rstrip(',')
            if ';' in line_for_detection and line_for_detection.count(';') > line_for_detection.count(','):
                long_column_list = line.replace(',', '.').split(';')
            else:
                long_column_list = line.split(',')

            for i in range(0, len(long_column_list), RECORD_SIZE):
                chunk = long_column_list[i : i + RECORD_SIZE]
                if any(cell.strip() for cell in chunk):
                    all_final_records.append(chunk)

    print(f"✅ Total record ditemukan: {len(all_final_records)}")

    # (Opsional) Bulatkan angka confidence
    for record in all_final_records:
        for idx in [3, 8, 10]:
            if idx < len(record):
                try:
                    record[idx] = f"{float(record[idx]):.4f}"
                except:
                    pass

    try:
        with open(output_filename, 'w', encoding='utf-8', newline='') as file_output:
            writer = csv.writer(file_output, delimiter=';', quoting=csv.QUOTE_MINIMAL)

            # Tulis header, dibungkus sebagai teks
            wrapped_header = [f'="{h}"' for h in correct_header]
            writer.writerow(wrapped_header)

            # Tulis setiap record, setiap nilai dibungkus sebagai teks
            for record in all_final_records:
                row = record + [""] * (RECORD_SIZE - len(record))
                wrapped_row = [f'="{cell.strip()}"' for cell in row]
                writer.writerow(wrapped_row)

        print("-" * 40)
        print("✅ File final berhasil disimpan sebagai:", output_filename)
        print("📌 Semua data ditulis sebagai teks (=\"...\") agar tidak diformat Excel.")
    except Exception as e:
        print(f"❌ Error saat menyimpan file: {e}")
