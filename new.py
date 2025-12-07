import lmdb
from PIL import Image
import io

lmdb_path = "textzoom/train1"

env = lmdb.open(lmdb_path, readonly=True, lock=False)

with env.begin(write=False) as txn:
    cursor = txn.cursor()

    for key, value in cursor:
        # value = raw JPEG/PNG bytes
        try:
            img = Image.open(io.BytesIO(value))
            img = img.convert("RGB")

            print("Key:", key)
            img.show()
        except Exception as e:
            print("Error:", e)

        break  # hapus jika ingin semua
