import os
import json
import requests
import io
import cv2
import numpy as np
from PIL import Image, ImageFilter
import mediapipe as mp
import cloudinary
import cloudinary.uploader
import argparse
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
CATALOG_PATH = "data/catalog.json"
PROGRESS_FILE = "data/_advanced_enhance_progress.json"

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

# ═══════════════════════════════════════════════════════════
# AI MODULES (Lazy Loaded)
# ═══════════════════════════════════════════════════════════
def get_red_mask(img_array):
    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 150, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    return mask

def get_hands_mask(img_array):
    mp_hands = mp.solutions.hands
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=4, min_detection_confidence=0.3) as hands:
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            h, w = img_array.shape[:2]
            for hand_landmarks in results.multi_hand_landmarks:
                points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark], dtype=np.int32)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, hull, 255)
                kernel = np.ones((25, 25), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def process_single_image(img_bytes, simple_lama):
    # Convert bytes to cv2 image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. Masking
    mask_red = get_red_mask(img_cv)
    mask_hands = get_hands_mask(img_cv)
    combined_mask = cv2.bitwise_or(mask_red, mask_hands)
    
    original_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # 2. Inpainting (only if defects found)
    if np.max(combined_mask) > 0:
        mask_pil = Image.fromarray(combined_mask).convert('L')
        inpainted_img = simple_lama(original_pil, mask_pil)
    else:
        inpainted_img = original_pil
        
    # 3. Background Removal
    from rembg import remove
    img_byte_arr = io.BytesIO()
    inpainted_img.save(img_byte_arr, format='PNG')
    subject_bytes = remove(img_byte_arr.getvalue())
    subject = Image.open(io.BytesIO(subject_bytes)).convert("RGBA")
    
    # 4. Crop and Studio Layout
    bbox = subject.getbbox()
    if bbox:
        subject = subject.crop(bbox)
        
    canvas_size = max(subject.width, subject.height) + 200
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (248, 248, 248, 255))
    
    x = (canvas_size - subject.width) // 2
    y = (canvas_size - subject.height) // 2
    
    shadow = Image.new("RGBA", subject.size, (0, 0, 0, 0))
    for sx in range(subject.width):
        for sy in range(subject.height):
            _, _, _, a = subject.getpixel((sx, sy))
            if a > 0:
                shadow.putpixel((sx, sy), (0, 0, 0, int(a * 0.35)))
                
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=25))
    canvas.paste(shadow, (x, y + 25), shadow)
    canvas.paste(subject, (x, y), subject)
    
    return canvas.convert("RGB")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"enhanced_ids": []}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Number of products to process")
    args = parser.parse_args()
    
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
        
    progress = load_progress()
    already_done = set(progress["enhanced_ids"])
    
    to_process = [p for p in catalog if p["id"] not in already_done and p.get("images")]
    to_process = to_process[:args.limit]
    
    if not to_process:
        print("✅ No products to process!")
        return

    # Load LaMa once
    from simple_lama_inpainting import SimpleLama
    simple_lama = SimpleLama()
    
    for product in to_process:
        print(f"\n📦 Processing [{product['id']}] {product.get('name')}")
        new_images = []
        all_ok = True
        
        for idx, img_url in enumerate(product['images']):
            print(f"  🖼️ Image {idx+1}/{len(product['images'])}...", end=" ", flush=True)
            try:
                # 1. Download
                resp = requests.get(img_url, timeout=20)
                if resp.status_code != 200:
                    raise Exception(f"HTTP {resp.status_code}")
                
                # 2. AI Processing
                final_img = process_single_image(resp.content, simple_lama)
                
                # 3. Upload to Cloudinary
                img_byte_arr = io.BytesIO()
                final_img.save(img_byte_arr, format='JPEG', quality=95)
                
                parts = img_url.split("/upload/")
                if len(parts) == 2:
                    path_part = parts[1]
                    if path_part.startswith("v"):
                        path_part = "/".join(path_part.split("/")[1:])
                    public_id = os.path.splitext(path_part)[0]
                else:
                    public_id = f"clothes_reseller/{product['id']}_{idx}"
                    
                result = cloudinary.uploader.upload(
                    img_byte_arr.getvalue(),
                    public_id=public_id,
                    overwrite=True,
                    resource_type="image",
                    invalidate=True,
                )
                new_images.append(result["secure_url"])
                print("✅ Enhanced & Uploaded!")
            except Exception as e:
                print(f"❌ Failed: {e}")
                new_images.append(img_url)
                all_ok = False
                
        # Save results
        product["images"] = new_images
        if all_ok:
            progress["enhanced_ids"].append(product["id"])
            save_progress(progress)
            
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
