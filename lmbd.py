import os
import lmdb
import cv2
from tqdm import tqdm

# -------------------------- CONFIG --------------------------
HR_DIR = "dataset_TextZoom/test/hard/HR"   # folder HR
LR_DIR = "dataset_TextZoom/test/hard/LR"   # folder LR
LMDB_OUT = "textzoom_lmdb/test/hard"       # hasil LMDB
# ------------------------------------------------------------


def extract_label(filename):
    """
    Format nama file: 0001__AB1626SZ.png
    Maka label = AB1626SZ
    """
    name = os.path.splitext(filename)[0]
    if "__" in name:
        return name.split("__")[1]
    return ""


def bytesize(img):
    """Encode image ke bytes PNG"""
    return cv2.imencode(".png", img)[1].tobytes()


def make_lmdb(hr_dir, lr_dir, out_path):
    # Pastikan folder output ada
    os.makedirs(out_path, exist_ok=True)

    # List file LR, urut numerik
    files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    env = lmdb.open(out_path, map_size=64 * 1024 * 1024)  # 64 MB
    txn = env.begin(write=True)

    idx = 1

    for f in tqdm(files, desc="Creating LMDB"):
        hr_path = os.path.join(hr_dir, f)
        lr_path = os.path.join(lr_dir, f)

        if not os.path.exists(hr_path):
            print(f"⚠ HR not found: {hr_path}")
            continue

        # Load images
        hr = cv2.imread(hr_path)
        lr = cv2.imread(lr_path)

        if hr is None or lr is None:
            print(f"⚠ Error reading: {f}")
            continue

        # Encode ke bytes
        hr_bin = bytesize(hr)
        lr_bin = bytesize(lr)

        # Label sebagai UTF-8 string (bukan pickle)
        label = extract_label(f)

        # Simpan ke LMDB
        txn.put(f"image_hr-{idx:09d}".encode(), hr_bin)
        txn.put(f"image_lr-{idx:09d}".encode(), lr_bin)
        txn.put(f"label-{idx:09d}".encode(), label.encode('utf-8'))

        idx += 1

        # Commit setiap 1000 file
        if idx % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)

    # Commit terakhir & simpan jumlah sampel
    txn.put("num-samples".encode(), str(idx - 1).encode())
    txn.commit()
    env.close()

    print(f"✅ LMDB created → {out_path}")
    print(f"Total samples: {idx - 1}")


if __name__ == "__main__":
    make_lmdb(HR_DIR, LR_DIR, LMDB_OUT)
