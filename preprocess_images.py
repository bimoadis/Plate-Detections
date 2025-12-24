# ==================================================
# 🖼️ Image Preprocessing with Auto-Orient, Resize, Grayscale, and Augmentation
# ==================================================
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import random
import sys

# ==================================================
# 🔹 Configuration
# ==================================================
OUTPUT_DIR = "img-prepro"  # Output folder untuk menyimpan setiap tahap
OUTPUTS_PER_IMAGE = 3  # Number of augmented outputs per training example
MAX_SIZE = 1280  # Maximum size for resize (fit within 1280x1280)
ROTATION_RANGE = (-8, 8)  # Rotation range in degrees
BRIGHTNESS_RANGE = (-20, 20)  # Brightness adjustment range in percentage
GRAYSCALE_PROBABILITY = 0.1  # 10% probability to apply grayscale in augmentation
NOISE_PIXEL_PERCENTAGE = 0.1  # Up to 0.1% of pixels for noise

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# 🔹 Auto-Orient: Apply EXIF orientation correction
# ==================================================
def auto_orient(img_path):
    """
    Auto-orient image based on EXIF data.
    Uses PIL's ImageOps.exif_transpose to handle orientation.
    Must read with PIL first to preserve EXIF data.
    """
    # Read with PIL to preserve EXIF data
    pil_img = Image.open(img_path)
    
    # Apply EXIF orientation correction
    pil_img = ImageOps.exif_transpose(pil_img)
    
    # Convert to RGB if needed
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    # Convert to OpenCV format (BGR)
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

# ==================================================
# 🔹 Resize: Fit within 1280x1280
# ==================================================
def resize_fit_within(img, max_size=1280):
    """
    Resize image to fit within max_size x max_size while maintaining aspect ratio.
    
    Args:
        img: Input image (BGR format)
        max_size: Maximum width or height
    
    Returns:
        Resized image
    """
    h, w = img.shape[:2]
    
    # Calculate scaling factor
    if w > h:
        if w > max_size:
            scale = max_size / w
        else:
            scale = 1.0
    else:
        if h > max_size:
            scale = max_size / h
        else:
            scale = 1.0
    
    # Resize if needed
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img.copy()
    
    return img_resized

# ==================================================
# 🔹 Grayscale: Convert to grayscale
# ==================================================
def apply_grayscale(img):
    """
    Convert image to grayscale.
    
    Args:
        img: Input image (BGR format)
    
    Returns:
        Grayscale image (still 3-channel BGR format for consistency)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert back to BGR (3 channels) for consistency
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray_bgr

# ==================================================
# 🔹 Rotation Augmentation: Between -8° and +8°
# ==================================================
def apply_rotation(img, angle):
    """
    Rotate image by specified angle.
    
    Args:
        img: Input image (BGR format)
        angle: Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        Rotated image
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation
    img_rotated = cv2.warpAffine(img, rotation_matrix, (new_w, new_h), 
                                  flags=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_REPLICATE)
    
    return img_rotated

# ==================================================
# 🔹 Brightness Augmentation: Between -20% and +20%
# ==================================================
def adjust_brightness(img, percentage):
    """
    Adjust image brightness by a percentage.
    
    Args:
        img: Input image (BGR format)
        percentage: Brightness adjustment percentage (-20 to +20)
    
    Returns:
        Brightness-adjusted image
    """
    # Convert percentage to factor (e.g., -20% = 0.80, +20% = 1.20)
    factor = 1.0 + (percentage / 100.0)
    
    # Multiply by factor and clip to valid range
    img_adjusted = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    
    return img_adjusted

# ==================================================
# 🔹 Noise Augmentation: Up to 0.1% of pixels
# ==================================================
def add_noise(img, pixel_percentage=0.1):
    """
    Add random noise to a percentage of pixels.
    
    Args:
        img: Input image (BGR format)
        pixel_percentage: Percentage of pixels to add noise (0.1 = 0.1%)
    
    Returns:
        Image with noise added
    """
    img_noisy = img.copy()
    h, w = img_noisy.shape[:2]
    total_pixels = h * w
    
    # Calculate number of pixels to modify
    num_noise_pixels = int(total_pixels * (pixel_percentage / 100.0))
    
    # Generate random pixel positions
    y_coords = np.random.randint(0, h, num_noise_pixels)
    x_coords = np.random.randint(0, w, num_noise_pixels)
    
    # Add random noise to selected pixels
    for i in range(num_noise_pixels):
        y, x = y_coords[i], x_coords[i]
        # Random noise value between -50 and +50
        noise = np.random.randint(-50, 51, 3)
        img_noisy[y, x] = np.clip(img_noisy[y, x].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img_noisy

# ==================================================
# 🔹 Main Preprocessing Pipeline
# ==================================================
def preprocess_image(img_path, output_dir):
    """
    Preprocess a single image with auto-orient, resize, grayscale,
    and generate multiple augmented versions. Save each step.
    
    Args:
        img_path: Path to input image
        output_dir: Directory to save all preprocessing steps
    
    Returns:
        True if successful, False otherwise
    """
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    ext = os.path.splitext(os.path.basename(img_path))[1]
    
    print(f"📸 Memproses: {os.path.basename(img_path)}")
    print("-" * 60)
    
    # Step 1: Auto-Orient (reads image and applies EXIF correction)
    print("🔄 Step 1: Auto-Orient...")
    try:
        img_orient = auto_orient(img_path)
    except Exception as e:
        print(f"⚠️ Gagal membaca atau orient: {img_path} - {str(e)}")
        return False
    
    if img_orient is None:
        print(f"⚠️ Gagal membaca: {img_path}")
        return False
    
    # Save Step 1: Auto-Orient
    step1_path = os.path.join(output_dir, f"{base_name}_01_auto_orient{ext}")
    cv2.imwrite(step1_path, img_orient)
    print(f"   ✅ Disimpan: {os.path.basename(step1_path)}")
    
    # Step 2: Resize (Fit within 1280x1280)
    print("🔄 Step 2: Resize (Fit within 1280x1280)...")
    img_resized = resize_fit_within(img_orient.copy(), MAX_SIZE)
    
    # Save Step 2: Resize
    step2_path = os.path.join(output_dir, f"{base_name}_02_resize{ext}")
    cv2.imwrite(step2_path, img_resized)
    h, w = img_resized.shape[:2]
    print(f"   ✅ Disimpan: {os.path.basename(step2_path)} (size: {w}x{h})")
    
    # Step 3: Grayscale
    print("🔄 Step 3: Grayscale...")
    img_grayscale = apply_grayscale(img_resized.copy())
    
    # Save Step 3: Grayscale
    step3_path = os.path.join(output_dir, f"{base_name}_03_grayscale{ext}")
    cv2.imwrite(step3_path, img_grayscale)
    print(f"   ✅ Disimpan: {os.path.basename(step3_path)}")
    
    # Step 4: Generate augmented versions (Total: 3 outputs per training example)
    print("🔄 Step 4: Augmentasi...")
    
    # Base preprocessed image (after grayscale)
    img_base = img_grayscale.copy()
    
    # Generate OUTPUTS_PER_IMAGE augmented versions
    for i in range(OUTPUTS_PER_IMAGE):
        img_aug = img_base.copy()
        aug_description = []
        step_counter = 0
        
        # Step 4.1: Apply rotation: Between -8° and +8°
        rotation_angle = random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1])
        img_aug = apply_rotation(img_aug, rotation_angle)
        aug_description.append(f"rot_{rotation_angle:+.1f}deg")
        
        # Save after rotation
        step_counter += 1
        step_rot_path = os.path.join(output_dir, f"{base_name}_04_aug_{i+1}_step{step_counter}_rotation_{rotation_angle:+.1f}deg{ext}")
        cv2.imwrite(step_rot_path, img_aug)
        print(f"   ✅ Step 4.{step_counter} - Rotation: {os.path.basename(step_rot_path)} ({rotation_angle:+.1f}°)")
        
        # Step 4.2: Apply grayscale to 10% of images (randomly)
        apply_gray = random.random() < GRAYSCALE_PROBABILITY
        if apply_gray:
            img_aug = apply_grayscale(img_aug)
            aug_description.append("gray")
            
            # Save after grayscale
            step_counter += 1
            step_gray_path = os.path.join(output_dir, f"{base_name}_04_aug_{i+1}_step{step_counter}_grayscale{ext}")
            cv2.imwrite(step_gray_path, img_aug)
            print(f"   ✅ Step 4.{step_counter} - Grayscale: {os.path.basename(step_gray_path)}")
        
        # Step 4.3: Apply brightness: Between -20% and +20%
        brightness_percentage = random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
        img_aug = adjust_brightness(img_aug, brightness_percentage)
        aug_description.append(f"bright_{brightness_percentage:+.1f}%")
        
        # Save after brightness
        step_counter += 1
        step_bright_path = os.path.join(output_dir, f"{base_name}_04_aug_{i+1}_step{step_counter}_brightness_{brightness_percentage:+.1f}%{ext}")
        cv2.imwrite(step_bright_path, img_aug)
        print(f"   ✅ Step 4.{step_counter} - Brightness: {os.path.basename(step_bright_path)} ({brightness_percentage:+.1f}%)")
        
        # Step 4.4: Apply noise: Up to 0.1% of pixels
        noise_percentage = random.uniform(0, NOISE_PIXEL_PERCENTAGE)
        img_aug = add_noise(img_aug, noise_percentage)
        aug_description.append(f"noise_{noise_percentage:.3f}%")
        
        # Save after noise (final augmented version)
        step_counter += 1
        aug_suffix = "_".join(aug_description)
        step_final_path = os.path.join(output_dir, f"{base_name}_04_aug_{i+1}_final_{aug_suffix}{ext}")
        cv2.imwrite(step_final_path, img_aug)
        print(f"   ✅ Step 4.{step_counter} - Final: {os.path.basename(step_final_path)}")
        print(f"      └─ Summary: Rotation: {rotation_angle:+.1f}°, "
              f"Grayscale: {'Yes' if apply_gray else 'No'}, "
              f"Brightness: {brightness_percentage:+.1f}%, "
              f"Noise: {noise_percentage:.3f}%")
        print()
    
    print("-" * 60)
    print(f"✅ Preprocessing selesai!")
    print(f"   📂 Semua hasil disimpan di: {os.path.abspath(output_dir)}")
    
    return True

# ==================================================
# 🔹 Process Single Image
# ==================================================
def main():
    # Get image path from command line argument or use default
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default: use first image from dataset_hr if exists, or ask user
        print("💡 Usage: python preprocess_images.py <path_to_image>")
        print("   Atau edit script untuk set default image path")
        print("-" * 60)
        
        # Try to find an image in dataset_hr
        from glob import glob
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(f"dataset_hr/{ext}"))
        
        if len(image_paths) > 0:
            img_path = image_paths[0]
            print(f"📸 Menggunakan gambar pertama dari dataset_hr: {os.path.basename(img_path)}")
        else:
            print("❌ Tidak ada gambar ditemukan. Silakan berikan path gambar sebagai argument.")
            print("   Contoh: python preprocess_images.py dataset_hr/image.png")
            return
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"❌ File tidak ditemukan: {img_path}")
        return
    
    print(f"📂 Input: {os.path.abspath(img_path)}")
    print(f"📂 Output folder: {os.path.abspath(OUTPUT_DIR)}")
    print(f"🔄 Outputs per image: {OUTPUTS_PER_IMAGE}")
    print(f"📐 Max size: {MAX_SIZE}x{MAX_SIZE}")
    print(f"🔄 Rotation range: {ROTATION_RANGE[0]}° to {ROTATION_RANGE[1]}°")
    print(f"💡 Brightness range: {BRIGHTNESS_RANGE[0]}% to {BRIGHTNESS_RANGE[1]}%")
    print(f"🎨 Grayscale probability: {GRAYSCALE_PROBABILITY * 100}%")
    print(f"🔊 Noise: Up to {NOISE_PIXEL_PERCENTAGE}% of pixels")
    print("-" * 60)
    
    # Process the image
    success = preprocess_image(img_path, OUTPUT_DIR)
    
    if not success:
        print("❌ Gagal memproses gambar")

if __name__ == "__main__":
    main()
