"""
Script untuk memperbaiki kompatibilitas basicsr dengan torchvision terbaru.
Memperbaiki import error: torchvision.transforms.functional_tensor
"""
import os
import sys
import importlib.util

def find_basicsr_degradations():
    """Mencari path file degradations.py di package basicsr"""
    try:
        import basicsr
        basicsr_path = os.path.dirname(basicsr.__file__)
        degradations_path = os.path.join(basicsr_path, 'data', 'degradations.py')
        
        if os.path.exists(degradations_path):
            return degradations_path
    except:
        pass
    
    # Coba cari di site-packages
    for path in sys.path:
        if 'site-packages' in path or 'dist-packages' in path:
            degradations_path = os.path.join(path, 'basicsr', 'data', 'degradations.py')
            if os.path.exists(degradations_path):
                return degradations_path
    
    return None

def fix_degradations_file(file_path):
    """Memperbaiki file degradations.py"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    
    print(f"📂 File ditemukan: {file_path}")
    
    # Baca isi file
    with open(file_path, "r", encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Perbaiki import yang rusak
    content = content.replace(
        "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
        "from torchvision.transforms.functional import rgb_to_grayscale"
    )
    
    # Fix fungsi yang pakai tensor version
    content = content.replace(
        "functional_tensor.rgb_to_grayscale",
        "rgb_to_grayscale"
    )
    
    # Cek apakah ada perubahan
    if content == original_content:
        print("ℹ️  File sudah benar, tidak perlu diperbaiki.")
        return False
    
    # Tulis ulang file
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Patch BERHASIL: basicsr/data/degradations.py telah diperbaiki untuk torchvision terbaru!")
    return True

def main():
    print("=" * 60)
    print("🔧 Fix BasicsR - TorchVision Compatibility")
    print("=" * 60)
    
    # Cari file degradations.py
    file_path = find_basicsr_degradations()
    
    if file_path is None:
        print("❌ File degradations.py tidak ditemukan!")
        print("   Pastikan basicsr sudah terinstall: pip install basicsr")
        return 1
    
    try:
        fix_degradations_file(file_path)
        print("\n✅ Selesai! Sekarang coba jalankan lr_to_sr.py lagi.")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())


