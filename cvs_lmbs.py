import lmdb
import six
from PIL import Image
import io
import os

def buf2PIL(imgbuf):
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

lmdb_path = "textzoom_custom_lmdb"  # ganti
save_dir = "exported"
os.makedirs(save_dir, exist_ok=True)

env = lmdb.open(lmdb_path, readonly=True, lock=False)
with env.begin() as txn:
    n = int(txn.get(b'num-samples'))
    print("Total samples:", n)

    for i in range(1, n+1):
        index = f"{i:09d}".encode()

        lr_key = b'image_lr-' + index
        hr_key = b'image_hr-' + index
        label_key = b'label-' + index

        # ambil data binary
        lr_bin = txn.get(lr_key)
        hr_bin = txn.get(hr_key)
        label = txn.get(label_key).decode()

        # convert ke gambar
        lr_img = buf2PIL(lr_bin)
        hr_img = buf2PIL(hr_bin)

        # save
        lr_img.save(f"{save_dir}/lr_{i:09d}_{label}.png")
        hr_img.save(f"{save_dir}/hr_{i:09d}_{label}.png")

        if i % 1000 == 0:
            print("Exported:", i)
