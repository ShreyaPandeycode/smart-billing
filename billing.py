# billing.py
import json
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

# load products if exists
PRODUCTS = {}
try:
    with open("products.json","r") as f:
        PRODUCTS = json.load(f)
except:
    PRODUCTS = {}

class Cart:
    def __init__(self):
        # key = sku_or_label -> {name, unit_price, qty}
        self.items = {}

    def add(self, sku, name, price, qty=1):
        if sku in self.items:
            self.items[sku]["qty"] += qty
        else:
            self.items[sku] = {"name": name, "unit_price": float(price), "qty": qty}

    def remove(self, sku):
        if sku in self.items:
            del self.items[sku]

    def clear(self):
        self.items = {}

    def summary(self):
        subtotal = 0.0
        lines = []
        for sku, it in self.items.items():
            line_total = it['unit_price'] * it['qty']
            lines.append({"sku": sku, "name": it['name'], "qty": it['qty'], "unit_price": it['unit_price'], "line_total": round(line_total,2)})
            subtotal += line_total
        tax = round(subtotal * 0.05, 2)
        total = round(subtotal + tax, 2)
        return {"lines": lines, "subtotal": round(subtotal,2), "tax": tax, "total": total}

def generate_invoice_pdf(cart:Cart, customer_name="Customer"):
    s = cart.summary()
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Smart Checkout - Invoice")
    y -= 28
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20
    c.drawString(40, y, f"Customer: {customer_name}")
    y -= 26
    for line in s['lines']:
        text = f"{line['name']} x{line['qty']}  ₹{line['unit_price']:.2f}  = ₹{line['line_total']:.2f}"
        c.drawString(40, y, text)
        y -= 16
        if y < 80:
            c.showPage(); y = 800
    y -= 8
    c.drawString(40, y, f"Subtotal: ₹{s['subtotal']:.2f}")
    y -= 16
    c.drawString(40, y, f"Tax (5%): ₹{s['tax']:.2f}")
    y -= 16
    c.drawString(40, y, f"Total: ₹{s['total']:.2f}")
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()
