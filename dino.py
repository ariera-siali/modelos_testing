import os
import cv2
import torch
import numpy as np
from pathlib import Path
import groundingdino
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
from PIL import Image, ImageDraw, ImageFont

# --- CONFIG ---
INPUT_DIR  = "fall_images"
OUTPUT_DIR = "results_dino"

# What to detect — separate with " . "
TEXT_PROMPT = "fallen person . not a fallen person"

# Thresholds — lower = more detections (but more false positives)
BOX_THRESHOLD  = 0.40   # confidence for bbox
TEXT_THRESHOLD = 0.35   # confidence for label match

# Model weights — downloaded automatically on first run
GDINO_PKG = Path(groundingdino.__file__).parent
MODEL_CONFIG  = str(GDINO_PKG / "config" / "GroundingDINO_SwinT_OGC.py")
MODEL_WEIGHTS = "weights/groundingdino_swint_ogc.pth"
WEIGHTS_URL   = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("weights", exist_ok=True)

# --- DOWNLOAD WEIGHTS IF NEEDED ---
if not os.path.exists(MODEL_WEIGHTS):
    print("Downloading GroundingDINO weights (~700MB)...")
    import urllib.request
    def progress(count, block_size, total_size):
        pct = int(count * block_size * 100 / total_size)
        print(f"\r  {pct}%", end="", flush=True)
    urllib.request.urlretrieve(WEIGHTS_URL, MODEL_WEIGHTS, reporthook=progress)
    print("\nWeights downloaded.")

# --- LOAD MODEL ---
print("Loading GroundingDINO model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = load_model(MODEL_CONFIG, MODEL_WEIGHTS, device=device)
model.eval()
print("Model loaded OK.\n")

# --- LABEL COLORS (BGR for OpenCV) ---
LABEL_COLORS = {
    "fallen person":   (0, 0, 255),    # red
    "person on ground":(0, 0, 255),    # red
    "forklift":        (0, 165, 255),  # orange
    "safety helmet":   (0, 200, 0),    # green
    "safety vest":     (0, 200, 0),    # green
    "default":         (255, 0, 255),  # magenta
}

def get_color(label):
    for key, color in LABEL_COLORS.items():
        if key in label.lower():
            return color
    return LABEL_COLORS["default"]

def draw_detections(image_path, boxes, logits, phrases, out_path):
    """Draw bboxes on the original image and save."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    MIN_SCORE = 0.38
    for box, score, phrase in zip(boxes, logits, phrases):
        # box is [cx, cy, w, h] normalized — convert to pixel xyxy
        if score.item() < MIN_SCORE:
            continue  # skip weak detections
        cx, cy, bw, bh = box.tolist()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        color = get_color(phrase)
        label = f"{phrase} {score:.2f}"

        # Draw box (3px)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Draw label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(out_path, img)

# --- PROCESS ---
image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}

if not os.path.exists(INPUT_DIR):
    print(f"Error: folder '{INPUT_DIR}' not found.")
    exit(1)

files = [f for f in os.listdir(INPUT_DIR)
         if Path(f).suffix.lower() in image_extensions]

if not files:
    print(f"No images found in '{INPUT_DIR}'.")
    exit(1)

print(f"Found {len(files)} image(s).")
print(f"Prompt: '{TEXT_PROMPT}'\n")

total_detections = 0

for filename in files:
    img_path = os.path.join(INPUT_DIR, filename)
    stem = Path(filename).stem

    try:
        # Load image for GroundingDINO
        image_source, image_tensor = load_image(img_path)

        # Run inference
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device,
        )

        n = len(boxes)
        total_detections += n

        if n > 0:
            out_path = os.path.join(OUTPUT_DIR, stem + "_dino.jpg")
            draw_detections(img_path, boxes, logits, phrases, out_path)
            labels = [f"{p}({s:.2f})" for p, s in zip(phrases, logits.tolist())]
            print(f"  ✓ {filename}: {n} detection(s) → {labels}")
        else:
            print(f"  – {filename}: nothing detected")

    except Exception as e:
        print(f"  ERROR on {filename}: {e}")

print(f"\n{'='*55}")
print(f"Done! {total_detections} total detections across {len(files)} images.")
print(f"Annotated images saved to '{OUTPUT_DIR}/'")
print()
print("Tip: if too many false positives → raise BOX_THRESHOLD to 0.40")
print("Tip: if missing detections    → lower BOX_THRESHOLD to 0.20")