# model_inference.py
"""
Model inference helpers for Smart Checkout.

Behavior:
 - If my_model.pt does not exist, the module will attempt to download it using:
   - direct HTTP(S) MODEL_URL (preferred) or Google Drive (gdown)
 - This file avoids raising on import due to missing OpenCV; instead it falls
   back to PIL-based image handling and annotations when possible.

Environment variables:
 - MODEL_URL  -> direct file URL or Google Drive share (use uc?export=download&id=... form)
 - HF_MODEL_FILENAME (optional) -> filename, default "my_model.pt"

Note: do NOT commit my_model.pt to GitHub; add it to .gitignore.
"""

import os
import re
import shutil
import traceback
import json
from pathlib import Path

# Try safe import of cv2; if unavailable, set cv2 = None and continue
try:
    import cv2
except Exception:
    cv2 = None
    print("[model_inference] WARNING: cv2 failed to import. Continuing without cv2.")
    traceback.print_exc()

# Common libs
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Barcode decoder (pyzbar) - optional
try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:
    zbar_decode = None
    print("[model_inference] WARNING: pyzbar not available; barcode decoding disabled.")

# Downloader
try:
    import gdown
except Exception:
    gdown = None

import requests

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "my_model.pt")
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
MODEL_URL = os.getenv("MODEL_URL")  # set this in Streamlit secrets

# -------------------------
# Download helpers
# -------------------------
def download_via_requests(url, dest_path, chunk_size=8192, timeout=120):
    """Download a file via requests streaming to dest_path."""
    try:
        print(f"[model_inference] Downloading from URL: {url}")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        return dest_path
    except Exception as e:
        print("[model_inference] HTTP download failed:", e)
        return None

def download_from_gdrive_with_gdown(url, dest_path):
    """Try to download a Google Drive URL using gdown."""
    if gdown is None:
        print("[model_inference] gdown not installed; skipping gdown method.")
        return None
    try:
        print(f"[model_inference] Using gdown to download: {url}")
        # gdown supports both share and uc links
        gdown.download(url, output=dest_path, quiet=False)
        if os.path.exists(dest_path):
            return dest_path
    except Exception as e:
        print("[model_inference] gdown download failed:", e)
    return None

def is_google_drive_url(url):
    return "drive.google.com" in (url or "")

def ensure_model_present():
    """Ensure MODEL_PATH exists. Try direct URL or gdown if MODEL_URL set.
       Returns the model path or None on failure."""
    if os.path.exists(MODEL_PATH):
        print(f"[model_inference] Model already present at: {MODEL_PATH}")
        return MODEL_PATH

    if not MODEL_URL:
        print("[model_inference] No MODEL_URL configured.")
        return None

    # prefer gdown for Google Drive
    if is_google_drive_url(MODEL_URL):
        got = download_from_gdrive_with_gdown(MODEL_URL, MODEL_PATH)
        if got and os.path.exists(got):
            return got

        # try to convert /d/<id>/ form into uc?export=download
        m = re.search(r'/d/([a-zA-Z0-9_-]+)', MODEL_URL)
        if m:
            file_id = m.group(1)
            uc = f"https://drive.google.com/uc?export=download&id={file_id}"
            got2 = download_via_requests(uc, MODEL_PATH)
            if got2 and os.path.exists(got2):
                return got2

        # try raw URL fallback
        got3 = download_via_requests(MODEL_URL, MODEL_PATH)
        if got3 and os.path.exists(got3):
            return got3

    else:
        # normal http/https link
        got = download_via_requests(MODEL_URL, MODEL_PATH)
        if got and os.path.exists(got):
            return got

    return None

# Attempt to ensure model is present (but do not raise here)
_model_path = ensure_model_present()
if _model_path:
    print(f"[model_inference] Model ready at: {_model_path}")
else:
    print(f"[model_inference] Model not present at {MODEL_PATH}. Will raise when inference is attempted.")

# -------------------------
# Load YOLO model (Ultralytics) if model exists and ultralytics is available
# -------------------------
model = None
if os.path.exists(MODEL_PATH):
    try:
        print(f"[model_inference] Loading YOLO model from: {MODEL_PATH}")
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        print("[model_inference] YOLO model loaded.")
    except Exception as e:
        print("[model_inference] Failed to load ultralytics YOLO model:", e)
        model = None
else:
    model = None

# -------------------------
# Optional product DB loader
# -------------------------
PRODUCTS = {}
try:
    if os.path.exists(os.path.join(BASE_DIR, "products.json")):
        with open(os.path.join(BASE_DIR, "products.json"), "r") as f:
            PRODUCTS = json.load(f)
except Exception:
    PRODUCTS = {}

# -------------------------
# Image loading helper (returns BGR numpy if cv2 available, else RGB numpy)
# -------------------------
def _load_image(image_input):
    """
    Load an image and return a numpy array.
    If cv2 available, returns BGR numpy array (as expected by YOLO).
    If cv2 is None, returns RGB numpy array (PIL -> numpy).
    Accepts: file path, bytes, file-like (Streamlit uploaded), or PIL.Image.
    """
    # file path
    if isinstance(image_input, str) and os.path.exists(image_input):
        if cv2 is not None:
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"OpenCV failed to read image: {image_input}")
            return img
        else:
            pil = Image.open(image_input).convert("RGB")
            return np.array(pil)  # RGB

    # file-like or bytes
    try:
        if hasattr(image_input, "read"):
            data = image_input.read()
        elif isinstance(image_input, (bytes, bytearray)):
            data = image_input
        else:
            data = None

        if data:
            if cv2 is not None:
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
                if img is not None:
                    return img
            # fallback to PIL
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            return np.array(pil)
    except Exception:
        pass

    # fallback: try PIL if input is PIL.Image
    try:
        if isinstance(image_input, Image.Image):
            pil = image_input.convert("RGB")
            return np.array(pil)
    except Exception:
        pass

    raise ValueError("Could not load image from provided input.")

# -------------------------
# Barcode decode helper
# -------------------------
def decode_barcode_from_crop(crop_np):
    """Accept crop as numpy array (BGR if cv2 available else RGB). Returns barcode string or None."""
    if zbar_decode is None:
        return None

    try:
        # If cv2 available, crop_np is BGR -> convert to grayscale via cv2
        if cv2 is not None:
            gray = cv2.cvtColor(crop_np, cv2.COLOR_BGR2GRAY)
            res = zbar_decode(gray)
            if res:
                return res[0].data.decode("utf-8")
            # try simple threshold
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            res = zbar_decode(th)
            if res:
                return res[0].data.decode("utf-8")
        else:
            # use PIL grayscale and pyzbar on PIL image
            pil = Image.fromarray(crop_np)  # crop_np is RGB
            res = zbar_decode(pil)
            if res:
                return res[0].data.decode("utf-8")
    except Exception:
        pass
    return None

# -------------------------
# Draw annotations using PIL (works whether cv2 exists or not)
# -------------------------
def draw_annotations(img_np, detections, line_thickness=6, font_size=28):
    """
    img_np: numpy array. If cv2 present it's BGR, otherwise RGB.
    Returns annotated image as numpy in BGR format if cv2 present, else RGB numpy.
    """
    # Convert to RGB for PIL drawing
    if cv2 is not None:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_np.copy()

    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d.get('bbox', (0,0,0,0))
        cls_name = str(d.get("cls_name", "object"))
        conf = d.get("conf", 0.0)
        label_txt = f"{cls_name} {conf:.2f}"

        bbox = draw.textbbox((0, 0), label_txt, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad = 8

        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=line_thickness)

        # label bg
        bg_x1 = x1
        bg_y1 = max(0, y1 - text_h - pad)
        bg_x2 = x1 + text_w + pad
        bg_y2 = y1
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill="black")
        draw.text((bg_x1 + 4, bg_y1 + 2), label_txt, fill="white", font=font)

        # barcode if present
        if d.get("barcode"):
            bc_txt = f"BC: {d['barcode']}"
            bc_bbox = draw.textbbox((0, 0), bc_txt, font=font)
            bc_w = bc_bbox[2] - bc_bbox[0]
            bc_h = bc_bbox[3] - bc_bbox[1]
            bc_bg_x1 = x1
            bc_bg_y1 = y2
            bc_bg_x2 = x1 + bc_w + pad
            bc_bg_y2 = y2 + bc_h + pad
            draw.rectangle([bc_bg_x1, bc_bg_y1, bc_bg_x2, bc_bg_y2], fill="yellow")
            draw.text((bc_bg_x1 + 4, bc_bg_y1 + 2), bc_txt, fill="black", font=font)

    out_np = np.array(pil)  # RGB
    if cv2 is not None:
        # convert back to BGR for downstream code expecting BGR
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    else:
        return out_np  # RGB

# -------------------------
# Main inference function
# -------------------------
def run_inference_on_image(image_input, conf_thresh=0.35, pad_ratio=0.06):
    """
    Run YOLO inference on an input image and return detections + annotated image.
    - image_input: file path OR bytes OR file-like object OR PIL Image
    - conf_thresh: confidence threshold to filter detections
    - pad_ratio: fraction of bbox width/height to expand crop for barcode detection
    Returns:
      - detections: list of dicts {'bbox':(x1,y1,x2,y2), 'conf', 'cls', 'cls_name', 'barcode'}
      - annotated: annotated image numpy (BGR if cv2 available else RGB)
    """
    # Ensure model is loaded
    global model
    if model is None:
        # try to download + load once more
        got = ensure_model_present()
        if got and os.path.exists(MODEL_PATH):
            try:
                from ultralytics import YOLO
                model = YOLO(MODEL_PATH)
                print("[model_inference] Model loaded on-demand.")
            except Exception as e:
                raise RuntimeError(f"Failed to load model at runtime: {e}") from e
        else:
            raise RuntimeError("Model not available. Set MODEL_URL in Streamlit secrets to a public download link.")

    # Load image (BGR if cv2 present, else RGB)
    img = _load_image(image_input)
    if img is None:
        raise ValueError("Could not load image input")

    # If cv2 is not present but ultralytics may expect BGR; convert accordingly by passing RGB
    # Ultralytics accepts numpy arrays in RGB or BGR depending on version — pass as-is and let it handle
    results = model(img, verbose=False)
    try:
        boxes = results[0].boxes
    except Exception:
        boxes = []

    h, w = (img.shape[0], img.shape[1]) if img is not None else (0, 0)
    detections = []
    for i in range(len(boxes)):
        try:
            xyxy = boxes[i].xyxy.cpu().numpy().squeeze().astype(int)
        except Exception:
            continue
        xmin, ymin, xmax, ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        conf = float(boxes[i].conf.item())
        if conf < conf_thresh:
            continue
        cls_id = int(boxes[i].cls.item())
        # model.names might be present
        try:
            cls_name = str(model.names.get(cls_id, cls_id))
        except Exception:
            cls_name = str(cls_id)

        # pad bbox to include barcode
        bw = xmax - xmin
        bh = ymax - ymin
        padx = int(bw * pad_ratio) + 5
        pady = int(bh * pad_ratio) + 5
        x1 = max(0, xmin - padx); y1 = max(0, ymin - pady)
        x2 = min(w, xmax + padx); y2 = min(h, ymax + pady)

        crop = img[y1:y2, x1:x2] if img is not None else None
        barcode = decode_barcode_from_crop(crop) if crop is not None else None

        det = {
            "bbox": (x1, y1, x2, y2),
            "conf": conf,
            "cls": cls_id,
            "cls_name": cls_name,
            "barcode": barcode
        }
        detections.append(det)

    annotated = draw_annotations(img, detections, line_thickness=6, font_size=28)
    return detections, annotated

# -------------------------
# Quick smoke test (optional)
# -------------------------
if __name__ == "__main__":
    test_path = os.path.join(BASE_DIR, "data", "images", "IMG_0026.jpg")
    if os.path.exists(test_path):
        dets, ann = run_inference_on_image(test_path)
        print("Detections:", dets)
        # save annotated image if cv2 available or via PIL
        if ann is not None:
            if cv2 is not None:
                cv2.imwrite("annotated_debug.jpg", ann)
            else:
                Image.fromarray(ann).save("annotated_debug.jpg")
            print("Saved annotated_debug.jpg")
    else:
        print("No test image at", test_path)
