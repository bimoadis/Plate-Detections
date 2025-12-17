import argparse

import cv2

import glob

import os

# Fix torchvision compatibility issue before importing basicsr
def fix_basicsr_torchvision():
    """Auto-fix basicsr compatibility with newer torchvision versions"""
    try:
        import basicsr
        basicsr_path = os.path.dirname(basicsr.__file__)
        degradations_path = os.path.join(basicsr_path, 'data', 'degradations.py')
        
        if os.path.exists(degradations_path):
            with open(degradations_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if fix is needed
            if 'from torchvision.transforms.functional_tensor import' in content:
                content = content.replace(
                    'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
                    'from torchvision.transforms.functional import rgb_to_grayscale'
                )
                content = content.replace(
                    'functional_tensor.rgb_to_grayscale',
                    'rgb_to_grayscale'
                )
                
                with open(degradations_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("✅ Auto-fixed basicsr torchvision compatibility")
    except Exception:
        pass  # Silently fail if fix cannot be applied

# Apply fix before importing
fix_basicsr_torchvision()

from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer


def main():
    """Inference demo for Real-ESRGAN.

    """
    parser = argparse.ArgumentParser()

    # Default paths
    default_input = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs')
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    
    parser.add_argument('-i', '--input', type=str, default=default_input, help='Input image or folder')

    parser.add_argument(

        '-n',

        '--model_name',

        type=str,

        default='RealESRGAN_x4plus',

        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '

              'realesr-animevideov3 | realesr-general-x4v3'))

    parser.add_argument('-o', '--output', type=str, default=default_output, help='Output folder')

    parser.add_argument(

        '-dn',

        '--denoise_strength',

        type=float,

        default=0.5,

        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '

              'Only used for the realesr-general-x4v3 model'))

    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')

    parser.add_argument(

        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')

    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')

    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')

    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')

    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')

    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')

    parser.add_argument(

        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')

    parser.add_argument(

        '--alpha_upsampler',

        type=str,

        default='realesrgan',

        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')

    parser.add_argument(

        '--ext',

        type=str,

        default='auto',

        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')

    parser.add_argument(

        '-g', '--gpu-id', type=int, default=0, help='gpu device to use (default=0) can be 0,1,2 for multi-gpu')
    
    parser.add_argument(
        '--max-size', type=int, default=0, 
        help='Maksimal dimensi gambar sebelum resize (0=disable, default: auto 2048 untuk CPU)')

    args = parser.parse_args()

    # Hanya menggunakan model net_g_5000
    print("=" * 60)
    print("🚀 Real-ESRGAN LR to SR Processing")
    print("=" * 60)
    print(f"📂 Input folder: {os.path.abspath(args.input)}")
    print(f"📂 Output folder: {os.path.abspath(args.output)}")
    print(f"🎮 GPU ID: {args.gpu_id}")
    print(f"📏 Scale: {args.outscale}x")
    print("-" * 60)
    
    # Model net_g_5000 menggunakan arsitektur RealESRGAN_x4plus
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    print("📋 Menggunakan arsitektur: RealESRGAN_x4plus (RRDBNet, scale=4)")
    
    # Tentukan path model - hanya net_g_5000.pth
    if args.model_path is not None:
        # Jika user memberikan path, pastikan itu net_g_5000
        if 'net_g_5000' not in args.model_path:
            print(f"⚠️  PERINGATAN: Model path yang diberikan bukan net_g_5000, akan menggunakan net_g_5000.pth dari weights folder")
            model_path = os.path.join('weights', 'net_g_latest.pth')
        else:
            model_path = args.model_path
            print(f"📦 Menggunakan model: {model_path}")
    else:
        # Gunakan net_g_5000.pth dari folder weights
        model_path = os.path.join('weights', 'net_g_latest.pth')
    
    # Cek apakah model file ada
    if not os.path.isfile(model_path):
        print(f"❌ ERROR: Model net_g_5000.pth tidak ditemukan di: {model_path}")
        print(f"   💡 Pastikan file net_g_5000.pth ada di folder 'weights'")
        return
    
    print(f"✅ Menggunakan model: {model_path}")

    # use dni to control the denoise strength
    dni_weight = None

    # restorer
    import torch
    
    # Force menggunakan GPU
    use_cuda = torch.cuda.is_available()
    device_name = 'CUDA' if use_cuda else 'CPU'
    
    if not use_cuda:
        print("❌ ERROR: CUDA tidak tersedia!")
        print("   💡 Script ini memerlukan GPU untuk berjalan.")
        print("   💡 Rekomendasi: Install PyTorch dengan CUDA support")
        print("      Contoh: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return
    
    # Pastikan GPU ID valid
    if args.gpu_id is not None and args.gpu_id >= torch.cuda.device_count():
        print(f"❌ ERROR: GPU ID {args.gpu_id} tidak tersedia!")
        print(f"   💡 GPU yang tersedia: 0-{torch.cuda.device_count()-1}")
        return
    
    # Set default GPU ID ke 0 jika tidak ditentukan
    if args.gpu_id is None:
        args.gpu_id = 0
    
    print(f"🎮 Menggunakan GPU: {args.gpu_id} ({torch.cuda.get_device_name(args.gpu_id)})")
    
    print(f"\n🔄 Memuat model ke memory...")
    print(f"   - Scale: {netscale}x")
    print(f"   - Precision: {'FP32' if args.fp32 else 'FP16'}")
    print(f"   - Device: {device_name}")
    
    # Auto-enable tile untuk gambar besar
    final_tile = args.tile
    print(f"   - Tile size: {final_tile if final_tile > 0 else 'Disabled'}")
    
    upsampler = RealESRGANer(

        scale=netscale,

        model_path=model_path,

        dni_weight=dni_weight,

        model=model,

        tile=final_tile,

        tile_pad=args.tile_pad,

        pre_pad=args.pre_pad,

        half=not args.fp32,

        gpu_id=args.gpu_id)
    
    print("✅ Model berhasil dimuat!\n")

    if args.face_enhance:  # Use GFPGAN for face enhancement

        from gfpgan import GFPGANer

        face_enhancer = GFPGANer(

            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',

            upscale=args.outscale,

            arch='clean',

            channel_multiplier=2,

            bg_upsampler=upsampler)

    # Buat folder input dan output jika belum ada
    os.makedirs(args.input, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):

        paths = [args.input]

    else:

        # Filter hanya file gambar
        all_paths = sorted(glob.glob(os.path.join(args.input, '*')))
        paths = [p for p in all_paths if os.path.isfile(p) and 
                 os.path.splitext(p)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]

    total_images = len(paths)
    
    if total_images == 0:
        print(f"❌ ERROR: Tidak ada gambar ditemukan di folder input!")
        print(f"   📂 Input folder: {os.path.abspath(args.input)}")
        print(f"   💡 Silakan tambahkan gambar (.jpg, .jpeg, .png, .bmp, .webp) ke folder tersebut")
        return
    
    print(f"📁 Ditemukan {total_images} gambar untuk diproses")
    print(f"📂 Input folder: {os.path.abspath(args.input)}")
    print(f"📂 Output folder: {os.path.abspath(args.output)}")
    print("-" * 60)
    
    import time
    start_time = time.time()

    for idx, path in enumerate(paths):

        imgname, extension = os.path.splitext(os.path.basename(path))

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"[{idx+1}/{total_images}] ❌ Gagal membaca: {imgname}")
            continue

        # Tampilkan info ukuran gambar
        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1
        img_size_mb = (h * w * channels) / (1024 * 1024)
        
        print(f"[{idx+1}/{total_images}] 🖼️  {imgname} ({w}x{h}, {img_size_mb:.1f}MB)...", end=" ", flush=True)

        if len(img.shape) == 3 and img.shape[2] == 4:

            img_mode = 'RGBA'

        else:

            img_mode = None

        # Auto-resize jika gambar terlalu besar (untuk mempercepat)
        max_dimension = args.max_size if args.max_size > 0 else 0
        original_img = img.copy()
        if max_dimension > 0 and max(h, w) > max_dimension:
            scale_factor = max_dimension / max(h, w)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"[Resized to {new_w}x{new_h}]", end=" ", flush=True)

        try:
            img_start = time.time()

            if args.face_enhance:

                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

            else:

                output, _ = upsampler.enhance(img, outscale=args.outscale)
            
            img_time = time.time() - img_start
            
            # Jika di-resize, scale up hasil ke ukuran asli yang diharapkan
            if max_dimension > 0 and max(h, w) > max_dimension:
                expected_h = int(h * args.outscale)
                expected_w = int(w * args.outscale)
                output = cv2.resize(output, (expected_w, expected_h), interpolation=cv2.INTER_CUBIC)
            
            print(f"✅ ({img_time:.1f}s)", end=" ", flush=True)

        except RuntimeError as error:

            print(f"❌ Error: {error}")
            print('   💡 Tip: Jika CUDA out of memory, coba set --tile dengan angka lebih kecil (contoh: -t 256)')
            continue

        else:

            if args.ext == 'auto':

                extension = extension[1:]

            else:

                extension = args.ext

            if img_mode == 'RGBA':  # RGBA images should be saved in png format

                extension = 'png'

            if args.suffix == '':

                save_path = os.path.join(args.output, f'{imgname}.{extension}')

            else:

                save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')

            cv2.imwrite(save_path, output)
            print(f"💾 Saved: {os.path.basename(save_path)}")
    
    total_time = time.time() - start_time
    avg_time = total_time / total_images if total_images > 0 else 0
    print("-" * 60)
    print(f"🎉 Selesai! {total_images} gambar diproses dalam {total_time:.1f} detik ({total_time/60:.1f} menit)")
    print(f"📊 Rata-rata: {avg_time:.1f} detik/gambar ({avg_time/60:.1f} menit/gambar)")
    
    # Tips optimasi jika lambat
    if avg_time > 30:  # Lebih dari 30 detik per gambar
        print("\n💡 TIPS UNTUK MEMPERCEPAT:")
        if args.tile == 0:
            print("   1. 🔲 Gunakan tile: Tambahkan -t 512 atau -t 256 untuk gambar besar")
        if avg_time > 60:
            print("   2. 📉 Kurangi ukuran gambar: Gunakan --max-size 1024 untuk resize otomatis")
            print("   3. ⚙️  Gunakan FP16: Pastikan --fp32 tidak diaktifkan (default FP16 lebih cepat)")
    
    print(f"📂 Hasil disimpan di: {os.path.abspath(args.output)}")


if __name__ == '__main__':

    main()

