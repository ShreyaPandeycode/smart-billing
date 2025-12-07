# app.py
import streamlit as st
# top of app.py
import traceback
try:
    import cv2
except Exception as e:
    cv2 = None
    print("WARNING: cv2 failed to import. Some features will be disabled.")
    traceback.print_exc()

import numpy as np
import json
from io import BytesIO
from datetime import datetime
from model_inference import run_inference_on_image
from tracker import SimpleTracker
from billing import Cart, generate_invoice_pdf
import qrcode
from PIL import Image

st.set_page_config(page_title="Smart Checkout", layout="wide")
st.title("🛒 Smart Checkout — Demo")

# load products db
try:
    with open("products.json","r") as f:
        PRODUCTS = json.load(f)
except:
    PRODUCTS = {}

# init session state
if "cart" not in st.session_state:
    st.session_state.cart = Cart()
if "seen_tracks" not in st.session_state:
    st.session_state.seen_tracks = set()
if "tracker" not in st.session_state:
    st.session_state.tracker = SimpleTracker(max_dist=90)

st.sidebar.header("Demo Controls")
conf_thresh = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, 0.35)
pad_ratio = st.sidebar.slider("BBox padding ratio", 0.0, 0.3, 0.06)

tab1, tab2, tab3 = st.tabs(["Scan & Detect", "Cart & Payment", "Admin"])

with tab1:
    st.header("Scan & Detect")
    st.write("Upload or use camera to take a photo of items.")
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    cam_btn = st.button("Use Camera (take photo)")
    img_input = None
    if uploaded:
        img_input = uploaded
    elif cam_btn:
        img_input = st.camera_input("Take a picture")
    if img_input is not None:
        with st.spinner("Running detection..."):
            dets, annotated = run_inference_on_image(img_input, conf_thresh=conf_thresh, pad_ratio=pad_ratio)
        if cv2 is None:
            st.error("OpenCV is not available on this server. Detection preview disabled.")
        else:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.write("Detections:", dets)
        # assign tracks
        dets = st.session_state.tracker.assign(dets)
        st.write("With track ids:", [{k:v for k,v in d.items() if k!='bbox'} for d in dets])
        if st.button("Add new detections to cart"):
            added = 0
            for d in dets:
                tid = d.get("track_id")
                if tid in st.session_state.seen_tracks:
                    continue
                # decide sku & price
                bc = d.get("barcode")
                if bc and bc in PRODUCTS:
                    sku = bc; name = PRODUCTS[bc]["name"]; price = PRODUCTS[bc]["price"]
                else:
                    sku = d.get("cls_name")
                    # fallback: if class name maps to product
                    if sku in PRODUCTS:
                        name = PRODUCTS[sku]["name"]; price = PRODUCTS[sku]["price"]
                    else:
                        name = sku; price = 30.0
                st.session_state.cart.add(sku, name, price, qty=1)
                st.session_state.seen_tracks.add(tid)
                added += 1
            st.success(f"Added {added} new item(s) to cart")

with tab2:
    st.header("Cart & Payment")
    summary = st.session_state.cart.summary()
    if summary['lines']:
        for line in summary['lines']:
            st.write(f"{line['name']} x{line['qty']} — ₹{line['line_total']:.2f}")
        st.write("---")
        st.write(f"Subtotal: ₹{summary['subtotal']:.2f}")
        st.write(f"Tax (5%): ₹{summary['tax']:.2f}")
        st.write(f"Total: ₹{summary['total']:.2f}")

        if st.button("Generate & Download Invoice PDF"):
            pdf_bytes = generate_invoice_pdf(st.session_state.cart, customer_name="Demo Customer")
            st.download_button("Download Invoice", data=pdf_bytes, file_name="invoice.pdf", mime="application/pdf")

        st.write("Generate UPI QR for payment (demo)")
        vpa = st.text_input("Enter UPI ID (vpa) for demo", value="your-vpa@bank")
        if st.button("Show UPI QR"):
            amount = summary['total']
            upi_link = f"upi://pay?pa={vpa}&pn=Shop&am={amount}&cu=INR&tn=SmartCheckout"
            qr = qrcode.make(upi_link)
            buf = BytesIO(); qr.save(buf, format="PNG"); buf.seek(0)
            st.image(buf)
            st.write("Scan with UPI app to pay the amount shown above.")
            st.info("This is a QR generator demo. Payment status must be confirmed manually or via backend webhook in real integration.")

        if st.button("Clear Cart"):
            st.session_state.cart.clear()
            st.session_state.seen_tracks.clear()
            st.success("Cart cleared")

    else:
        st.info("Cart is empty. Add items from Scan & Detect tab.")

with tab3:
    st.header("Admin")
    st.subheader("Products loaded")
    st.write(PRODUCTS)
    st.subheader("Session cart")
    st.write(st.session_state.cart.items)
    st.subheader("Seen track ids")
    st.write(list(st.session_state.seen_tracks))
