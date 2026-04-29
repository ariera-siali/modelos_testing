import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import torchvision.transforms as T
import torch

# --- CONFIG ---
MODEL_ID   = "OpenGVLab/InternVL2-2B"
INPUT_DIR  = "images"
OUTPUT_DIR = "results_internvl"
DEBUG      = True   # Set False to silence raw model output

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).cuda().eval()
print("Model loaded OK.\n")

# --- IMAGE LOADER ---
def load_image_tensor(image_path, input_size=448):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    return transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

def chat(pixel_values, question, max_new_tokens=512):
    result = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config=dict(max_new_tokens=max_new_tokens, do_sample=False),
    )
    return result[0] if isinstance(result, (list, tuple)) else result

# --- WHAT TO DETECT ---
TARGETS = ["forklift", "helmet", "safety vest", "fallen person"]

LABEL_COLORS = {
    "forklift":      "#FF4444",
    "helmet":        "#44CC44",
    "safety vest":   "#44CC44",
    "vest":          "#44CC44",
    "fallen":        "#FF8800",
    "fallen person": "#FF8800",
    "worker":        "#4488FF",
    "person":        "#4488FF",
    "default":       "#FFFF00",
}

def get_color(label):
    for key, color in LABEL_COLORS.items():
        if key in label.lower():
            return color
    return LABEL_COLORS["default"]

# --- BBOX PARSING ---
def parse_boxes(text, img_w, img_h):
    """
    Parse InternVL2 grounding output.
    Handles formats:
      <ref>label</ref><box>[[x1,y1,x2,y2]]</box>
      <box>[[x1,y1,x2,y2]]</box>
    Coords normalised to 0-1000.
    """
    detections = []

    # Format 1: <ref>label</ref><box>[[x1,y1,x2,y2]]</box>
    pattern = r'<ref>(.*?)</ref>\s*<box>\[\[([\d\s,]+)\]\]</box>'
    for label, coords_str in re.findall(pattern, text):
        try:
            coords = [float(v.strip()) for v in coords_str.split(',')]
            if len(coords) == 4:
                x1 = int(coords[0] / 1000 * img_w)
                y1 = int(coords[1] / 1000 * img_h)
                x2 = int(coords[2] / 1000 * img_w)
                y2 = int(coords[3] / 1000 * img_h)
                detections.append((label.strip(), x1, y1, x2, y2))
        except Exception:
            continue

    # Format 2: plain [[x1,y1,x2,y2]] (no ref tag)
    if not detections:
        for coords_str in re.findall(r'\[\[([\d\s,]+)\]\]', text):
            try:
                coords = [float(v.strip()) for v in coords_str.split(',')]
                if len(coords) == 4:
                    x1 = int(coords[0] / 1000 * img_w)
                    y1 = int(coords[1] / 1000 * img_h)
                    x2 = int(coords[2] / 1000 * img_w)
                    y2 = int(coords[3] / 1000 * img_h)
                    detections.append(("object", x1, y1, x2, y2))
            except Exception:
                continue

    return detections

def draw_boxes(image_path, detections, out_path):
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for (label, x1, y1, x2, y2) in detections:
        color = get_color(label)
        for i in range(3):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)
        tb = draw.textbbox((x1, y1), label, font=font)
        tw, th = tb[2]-tb[0], tb[3]-tb[1]
        draw.rectangle([x1, y1-th-4, x1+tw+4, y1], fill=color)
        draw.text((x1+2, y1-th-2), label, fill="white", font=font)

    img.save(out_path)
    return out_path

# --- PROCESS ---
image_extensions = ('.png', '.jpg', '.jpeg', '.webp')

if not os.path.exists(INPUT_DIR):
    print(f"Error: folder '{INPUT_DIR}' not found.")
    exit(1)

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]
if not files:
    print(f"No images found in '{INPUT_DIR}'.")
    exit(1)

print(f"Found {len(files)} image(s). Processing...\n")

for filename in tqdm(files):
    img_path = os.path.join(INPUT_DIR, filename)
    stem     = os.path.splitext(filename)[0]
    print(f"\n{'='*60}")
    print(f"Image: {filename}")

    try:
        pixel_values = load_image_tensor(img_path)
        orig_img     = Image.open(img_path).convert("RGB")
        img_w, img_h = orig_img.size

        # ── PASS 1: plain description ─────────────────────────────
        q1 = (
            "<image>\n"
            "Look carefully at this image. "
            "Are there any of the following: forklifts, workers wearing helmets, "
            "workers wearing safety vests, or fallen/injured people on the ground? "
            "List exactly what you see from those categories. Be specific."
        )
        description = chat(pixel_values, q1, max_new_tokens=200)
        if DEBUG:
            print(f"  [Pass 1] {description}")

        # Save description
        with open(os.path.join(OUTPUT_DIR, stem + ".txt"), "w") as f:
            f.write(description)

        # Check if anything relevant was detected
        desc_lower = description.lower()
        detected_labels = [t for t in TARGETS
                           if t in desc_lower
                           and "no " + t not in desc_lower
                           and "not" not in desc_lower.split(t)[0][-15:]]

        if not detected_labels:
            print(f"  → Nothing relevant detected, skipping grounding.")
            continue

        print(f"  → Detected: {detected_labels}. Running grounding pass...")

        # ── PASS 2: grounding for each detected label ─────────────
        all_detections = []
        for label in detected_labels:
            q2 = (
                f"<image>\n"
                f"Locate all {label}s in this image. "
                f"Output each one as: <ref>{label}</ref><box>[[x1,y1,x2,y2]]</box> "
                f"where coordinates are integers normalized to 0-1000. "
                f"Output only the tags, nothing else."
            )
            grounding_resp = chat(pixel_values, q2, max_new_tokens=256)
            if DEBUG:
                print(f"  [Pass 2 – {label}] raw: {grounding_resp}")

            boxes = parse_boxes(grounding_resp, img_w, img_h)
            if boxes:
                all_detections.extend(boxes)
                print(f"  → {label}: {len(boxes)} box(es) found")
            else:
                print(f"  → {label}: no valid boxes parsed from response")

        # ── DRAW ─────────────────────────────────────────────────
        if all_detections:
            out_img_path = os.path.join(OUTPUT_DIR, stem + "_bbox.jpg")
            draw_boxes(img_path, all_detections, out_img_path)
            print(f"  ✓ Saved annotated image: {stem}_bbox.jpg")
        else:
            print(f"  ✗ Grounding returned no parseable boxes.")

    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*60}")
print(f"Done! Results in '{OUTPUT_DIR}/'")
print(f"  *_bbox.jpg  → annotated images")
print(f"  *.txt       → text descriptions")