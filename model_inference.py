# model_inference.py
"""
Model inference helpers for Smart Checkout.

Auto-download behavior:
 - If my_model.pt does not exist, the module will attempt to download it using:
   1) Hugging Face model hub (if HF_MODEL_REPO set)
   2) Direct HTTP(S) URL from MODEL_URL env var
   3) Google Drive link via gdown or a requests fallback

Environment variables (set in Streamlit secrets or env):
 - MODEL_URL           -> direct file URL or Google Drive share (use uc?export=download id form)
 - HF_MODEL_REPO       -> huggingface repo id, e.g. "username/repo-name" (optional)
 - HF_MODEL_FILENAME   -> filename in HF repo (default: my_model.pt)
 - HUGGINGFACE_TOKEN   -> HF token if repo is private (optional)

Make sure to add "my_model.pt" to .gitignore so you don't push the model to GitHub.
"""

import os
import re
import sys
import time
import json
import shutil
import requests
from pathlib import Path


# near top of model_inference.py
try:
    import cv2
except Exception as e:
    cv2 = None
    print("WARNING: cv2 failed to import inside model_inference.")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "my_model.pt")
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

HF_REPO = os.getenv("HF_MODEL_REPO")  # e.g. "your-username/smart-checkout-model"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
MODEL_URL = os.getenv("MODEL_URL")  # fallback direct URL (HTTP/HTTPS or Google Drive uc link)

# -------------------------
# Download helpers
# -------------------------
def download_from_hf(repo_id, filename, token=None, dest_path=MODEL_PATH):
    """Download a file from a huggingface model repo using huggingface_hub.hf_hub_download"""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        print("huggingface_hub not installed:", e)
        return None
    try:
        print(f"[model_inference] Downloading {filename} from HF repo {repo_id} ...")
        local = hf_hub_download(repo_id=repo_id, filename=filename, use_auth_token=token)
        # copy to desired dest_path if different
        if local != dest_path:
            shutil.copy(local, dest_path)
        return dest_path
    except Exception as e:
        print("[model_inference] Hugging Face download failed:", e)
        return None

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
    """Try to download Google Drive file using gdown module (recommended)."""
    try:
        import gdown
    except Exception as e:
        print("[model_inference] gdown not installed:", e)
        return None
    try:
        print(f"[model_inference] Using gdown to download: {url}")
        gdown.download(url, output=dest_path, quiet=False)
        if os.path.exists(dest_path):
            return dest_path
        return None
    except Exception as e:
        print("[model_inference] gdown download failed:", e)
        return None

def is_google_drive_url(url):
    return "drive.google.com" in (url or "")

def ensure_model_present():
    """Ensure MODEL_PATH exists. Try HF -> direct URL -> gdown -> requests fallback."""
    if os.path.exists(MODEL_PATH):
        print(f"[model_inference] Model already present at: {MODEL_PATH}")
        return MODEL_PATH

    # 1) Hugging Face
    if HF_REPO:
        got = download_from_hf(HF_REPO, MODEL_FILENAME, token=HF_TOKEN)
        if got and os.path.exists(got):
            print("[model_inference] Downloaded model from Hugging Face.")
            return got

    # 2) Direct URL (HTTP/S)
    if MODEL_URL:
        # If it's a Google Drive link, prefer gdown
        if is_google_drive_url(MODEL_URL):
            # Try gdown first
            got = download_from_gdrive_with_gdown(MODEL_URL, MODEL_PATH)
            if got and os.path.exists(got):
                return got
            # fallback: try to convert to uc?export=download form if it's not already
            m = re.search(r'/d/([a-zA-Z0-9_-]+)', MODEL_URL)
            if m:
                file_id = m.group(1)
                uc = f"https://drive.google.com/uc?export=download&id={file_id}"
                got2 = download_via_requests(uc, MODEL_PATH)
                if got2 and os.path.exists(got2):
                    return got2
            # final fallback: try the raw URL with requests
            got3 = download_via_requests(MODEL_URL, MODEL_PATH)
            if got3 and os.path.exists(got3):
                return got3
        else:
            # normal http(s) link
            got = download_via_requests(MODEL_URL, MODEL_PATH)
            if got and os.path.exists(got):
                return got

    # If we reach here, we failed to obtain the model
    return None

# -------------------------
# Ensure model file exists (download if missing)
# -------------------------
if not os.path.exists(MODEL_PATH):
    print(f"[model_inference] Model not found at {MODEL_PATH}. Attempting download...")
    got = ensure_model_present()
    if not got or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH} and automatic download failed.\n"
            "Deployment instructions: upload the model to a public HTTP URL, Google Drive (shareable), "
            "or Hugging Face model hub. Then set MODEL_URL (or HF_MODEL_REPO and HUGGINGFACE_TOKEN) "
            "in Streamlit Cloud secrets."
        )

# -------------------------
# Load YOLO model (Ultralytics)
# -------------------------
try:
    print("[model_inference] Loading YOLO model from:", MODEL_PATH)
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print("[model_inference] Model loaded.")
except Exception as e:
    print("[model_inference] Failed to load ultralytics YOLO model:", e)
    raise

# -------------------------
# Optional product DB loader
# -------------------------
PRODUCTS = {}
if os.path.exists(os.path.join(BASE_DIR, "products.json")):
    try:
        with open(os.path.join(BASE_DIR, "products.json"), "r") as f:
            PRODUCTS = json.load(f)
    except Exception:
        PRODUCTS = {}

# -------------------------
# Image loading helper
# -------------------------
def _load_image(image_input):
    """
    Load an image into BGR OpenCV numpy array.
    Accepts: file path (str), bytes, file-like (Streamlit UploadedFile), or PIL Image.
    """
    # file path
    if isinstance(image_input, (str,)) and os.path.exists(image_input):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"OpenCV failed to read image: {image_input}")
        return img

    # bytes or file-like
    try:
        if hasattr(image_input, "read"):
            data = image_input.read()
        elif isinstance(image_input, (bytes, bytearray)):
            data = image_input
        else:
            data = None

        if data:
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass

    # fallback to PIL
    try:
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        pil = Image.open(image_input).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError("Could not load image: " + str(e))

# -------------------------
# Barcode decode helper
# -------------------------
def decode_barcode_from_crop(crop_bgr):
    try:
        from pyzbar.pyzbar import decode as zbar_decode
    except Exception:
        print("[model_inference] pyzbar not installed or failed to import.")
        return None

    try:
       if cv2 is None:
             return None
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        res = zbar_decode(gray)
        if res:
            return res[0].data.decode("utf-8")
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res = zbar_decode(th)
        if res:
            return res[0].data.decode("utf-8")
    except Exception:
        pass
    return None

# -------------------------
# Draw annotations (PIL) - large readable labels
# -------------------------
def draw_annotations(img_bgr, detections, line_thickness=6, font_size=28):
     if cv2 is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)

    # Try common fonts, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        cls_name = str(d.get("cls_name", "object"))
        conf = d.get("conf", 0.0)
        label_txt = f"{cls_name} {conf:.2f}"

        bbox = draw.textbbox((0, 0), label_txt, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad = 8

        draw.rectangle([x1, y1, x2, y2], outline="red", width=line_thickness)

        bg_x1 = x1
        bg_y1 = max(0, y1 - text_h - pad)
        bg_x2 = x1 + text_w + pad
        bg_y2 = y1
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill="black")
        draw.text((bg_x1 + 4, bg_y1 + 2), label_txt, fill="white", font=font)

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

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# -------------------------
# Main inference function
# -------------------------
def run_inference_on_image(image_input, conf_thresh=0.35, pad_ratio=0.06):
    img = _load_image(image_input)
    if img is None:
        raise ValueError("Could not load image input")

    h, w = img.shape[:2]

    # Run model
    results = model(img, verbose=False)
    try:
        boxes = results[0].boxes
    except Exception:
        boxes = []

    detections = []
    for i in range(len(boxes)):
        try:
            xyxy = boxes[i].xyxy.cpu().numpy().squeeze().astype(int)
        except Exception:
            continue
        xmin, ymin, xmax, ymax = xyxy
        conf = float(boxes[i].conf.item())
        if conf < conf_thresh:
            continue
        cls_id = int(boxes[i].cls.item())
        cls_name = str(model.names.get(cls_id, cls_id))

        bw = xmax - xmin
        bh = ymax - ymin
        padx = int(bw * pad_ratio) + 5
        pady = int(bh * pad_ratio) + 5
        x1 = max(0, xmin - padx); y1 = max(0, ymin - pady)
        x2 = min(w, xmax + padx); y2 = min(h, ymax + pady)

        crop = img[y1:y2, x1:x2]
        barcode = decode_barcode_from_crop(crop)

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
        cv2.imwrite("annotated_debug.jpg", ann)
        print("Saved annotated_debug.jpg")
    else:
        print("No test image at", test_path)
