import os
import csv

hr_dir = "dataset_TextZoom/test/medium/HR"
lr_dir = "dataset_TextZoom/test/medium/LR"
output_csv = "annotation_medium.csv"

rows = []
for filename in sorted(os.listdir(hr_dir)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # Contoh nama file: 0001__AB1345HB.png
    base_name = filename.split('.')[0]            # 0001__AB1345HB
    parts = base_name.split("__")

    if len(parts) != 2:
        print("Format filename salah:", filename)
        continue

    label = parts[1]                              # AB1345HB

    hr_path = f"hr/{filename}"
    lr_path = f"lr/{filename}"

    rows.append([hr_path, lr_path, label])

# Tulis ke CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["HR_path", "LR_path", "label"])
    writer.writerows(rows)

print("CSV berhasil dibuat:", output_csv)
