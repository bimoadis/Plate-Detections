"""
Script untuk mengecek apakah PyTorch dengan CUDA sudah terinstall dan berfungsi
"""
import sys

print("=" * 60)
print("🔍 Checking PyTorch CUDA Installation")
print("=" * 60)

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # Cek apakah CUDA tersedia
    cuda_available = torch.cuda.is_available()
    print(f"🔧 CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"📊 Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n🖥️  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # Test CUDA dengan tensor kecil
        try:
            x = torch.randn(3, 3).cuda()
            print(f"\n✅ CUDA Test: SUCCESS - Tensor berhasil dibuat di GPU")
        except Exception as e:
            print(f"\n❌ CUDA Test: FAILED - {e}")
    else:
        print("\n⚠️  CUDA tidak tersedia!")
        print("   Kemungkinan penyebab:")
        print("   1. GPU tidak didukung")
        print("   2. Driver NVIDIA tidak terinstall")
        print("   3. PyTorch versi CPU terinstall (bukan CUDA)")
        print("\n💡 Untuk install PyTorch dengan CUDA:")
        print("   pip uninstall torch torchvision")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
except ImportError:
    print("❌ PyTorch tidak terinstall!")
    print("   Install dengan: pip install torch torchvision")
    sys.exit(1)

print("\n" + "=" * 60)
