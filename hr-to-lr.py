from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random
import os
from glob import glob
import io

hr_dir = 'dataset_hr'
lr_dir = 'dataset_lr_like_example'
os.makedirs(lr_dir, exist_ok=True)

def downscale_image(img, target_height=20):
    """Downscale gambar ke resolusi sangat kecil seperti contoh"""
    w, h = img.size
    # Target height sekitar 20-30px seperti contoh
    ratio = target_height / h
    new_w = max(1, int(w * ratio))
    new_h = target_height
    return img.resize((new_w, new_h), Image.BILINEAR)

def add_blocky_upscale(img, factor=2):
    """Downscale -> Upscale untuk membuat pixelated blocky artifact"""
    w, h = img.size
    small = img.resize((max(1, w//factor), max(1, h//factor)), Image.BILINEAR)
    return small.resize((w, h), Image.NEAREST)

def add_gaussian_noise(img, std=2):
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, np_img.shape)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

def increase_contrast(img, amount=1.5):
    np_img = np.array(img).astype(np.float32)
    np_img = (np_img - 128) * amount + 128
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

def adjust_brightness(img, factor=0.9):
    """Membuat gambar sedikit lebih gelap seperti contoh"""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def apply_jpeg_artifacts(img, quality=25):
    """Tambah JPEG compression artifacts"""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

for hr_path in glob(f"{hr_dir}/*.png") + glob(f"{hr_dir}/*.jpg") + glob(f"{hr_dir}/*.jpeg"):
    img = Image.open(hr_path).convert("RGB")
    
    # 1. Downscale ke resolusi sangat kecil (seperti contoh ~20-35px tinggi)
    target_h = random.randint(18, 35)
    img = downscale_image(img, target_height=target_h)
    
    # 2. Blur ringan
    blur_strength = random.uniform(0.3, 1.0)
    img = img.filter(ImageFilter.GaussianBlur(blur_strength))
    
    # 3. Blocky/pixelated effect
    if random.random() > 0.3:
        img = add_blocky_upscale(img, factor=random.choice([2, 3]))
    
    # 4. Noise sangat tipis
    img = add_gaussian_noise(img, std=random.uniform(1.0, 4.0))
    
    # 5. Tingkatkan kontras (hitam-putih lebih tegas seperti contoh)
    img = increase_contrast(img, amount=random.uniform(1.3, 1.8))
    
    # 6. Brightness sedikit gelap (seperti contoh)
    img = adjust_brightness(img, factor=random.uniform(0.85, 1.0))
    
    # 7. JPEG artifact kasar seperti contoh
    jpeg_quality = random.randint(15, 40)
    img = apply_jpeg_artifacts(img, quality=jpeg_quality)
    
    # 8. Simpan sebagai PNG (untuk mempertahankan hasil akhir)
    base_name = os.path.splitext(os.path.basename(hr_path))[0]
    out_path = os.path.join(lr_dir, f"{base_name}.png")
    img.save(out_path)
    
    print(f"Generated LR: {out_path} (size: {img.size[0]}x{img.size[1]})")

print("\n✅ Selesai! LR sekarang mirip contoh gambar plat nomor low-res.")
print(f"📁 Output disimpan di: {lr_dir}")
