# app.py (FINAL FINAL FIXED)
import streamlit as st
import pandas as pd
import easyocr
from PIL import Image
import numpy as np
import io
import re
from rapidfuzz import process, fuzz
from fpdf import FPDF
from datetime import datetime

st.set_page_config(layout="wide", page_title="Kirana Bill Scanner")

st.title("Kirana Bill Scanner — Overcharge Detector")
st.markdown("Upload a photo of the kirana bill (printed or handwritten).")


# --------------------------
# Load MRP DB
# --------------------------
@st.cache_data
def load_mrp(path="mrp.csv"):
    df = pd.read_csv(path)
    df['item_lower'] = df['item'].str.lower()
    return df

mrp_df = load_mrp()


# --------------------------
# EasyOCR reader
# --------------------------
@st.cache_resource
def get_reader(lang_list=["en"]):
    return easyocr.Reader(lang_list, gpu=False)

reader = get_reader()

uploaded = st.file_uploader("Upload bill image (JPG / PNG)", type=['png','jpg','jpeg'])

# --------------------------
# Patterns
# --------------------------
price_pattern = re.compile(r'₹?\s*([0-9]+(?:[.,][0-9]{1,2})?)\b')
qty_pattern = re.compile(r'([0-9]+(?:[.,][0-9]*)?)\s?(kg|g|ltr|l|pcs|pc|pkt|pack)?', re.IGNORECASE)

IGNORE_WORDS = {"bill", "product", "quantity", "price", "unit", "total", "subtotal", "gst", "tax", "amount"}

def to_number(s):
    if s is None:
        return None
    s = str(s).replace('₹','').replace(',','').strip()
    try:
        return float(s)
    except:
        return None


def get_best_match(item, mrp_df, score_cutoff=60):
    if not item or item.strip()=="":
        return None, 0
    choices = mrp_df['item_lower'].tolist()
    match = process.extractOne(item.lower(), choices, scorer=fuzz.token_sort_ratio)
    if match is None:
        return None, 0
    matched_str, score, idx = match
    if score < score_cutoff:
        return None, score
    row = mrp_df[mrp_df['item_lower']==matched_str].iloc[0]
    return row.to_dict(), score


# --------------------------
# OCR extraction
# --------------------------
def extract_text_from_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    arr = np.array(img)
    ocr_result = reader.readtext(arr, detail=1, paragraph=False)

    lines = []
    for bbox, text, conf in ocr_result:
        ys = [pt[1] for pt in bbox]
        ycenter = sum(ys)/len(ys)
        lines.append((ycenter, text, conf))

    lines_sorted = [t for _, t, _ in sorted(lines, key=lambda x: x[0])]
    cleaned = [re.sub(r'\s+', ' ', s).strip() for s in lines_sorted]
    return cleaned, img



# --------------------------
# Parsing
# --------------------------
def parse_lines_horizontal(lines):
    rows = []
    used = set()

    for i, line in enumerate(lines):
        if i in used:
            continue

        text = line.strip()
        if not text:
            continue

        low = re.sub(r'[^a-zA-Z0-9 ]',' ', text).strip().lower()
        if low in IGNORE_WORDS:
            continue

        prices = price_pattern.findall(text)
        price_val = None
        item_guess = None
        paired_from_next = False

        if prices:
            price_val = to_number(prices[-1])
            item_guess = price_pattern.sub('', text).strip()
        else:
            if i+1 < len(lines):
                nxt = lines[i+1].strip()
                nxt_prices = price_pattern.findall(nxt)
                if nxt_prices and re.search(r'[A-Za-z]', text):
                    price_val = to_number(nxt_prices[-1])
                    item_guess = text
                    used.add(i+1)
                    paired_from_next = True

        if item_guess:
            item_guess = re.sub(r'[:\-–—]+', ' ', item_guess).strip()
            if re.fullmatch(r'[\d\W_]+', item_guess):
                item_guess = None

        if (price_val is None) and (item_guess is None):
            continue

        qty_match = qty_pattern.search(item_guess) if item_guess else None

        if item_guess:
            item_guess = re.sub(r'\b\d{1,4}\b$', '', item_guess).strip()

        if item_guess and re.search(r'\b(total|subtotal|gst|tax|balance)\b', item_guess, re.IGNORECASE):
            continue

        match_row, score = get_best_match(item_guess or "", mrp_df)
        matched_item = match_row['item'] if match_row else ""
        mrp = match_row['mrp'] if match_row else None
        unit = match_row['unit'] if match_row else ""
        overcharge = None
        status = ""

        if (price_val is not None) and (mrp is not None):
            try:
                overcharge = round(float(price_val) - float(mrp), 2)
                status = "Overcharged" if overcharge > 0 else "OK"
            except:
                status = ""

        if re.search(r'\bmisc\b|\bother\b|\bextra\b|\bround\b', text, re.IGNORECASE):
            status = status + ("; " if status else "") + "Suspicious Misc Charge"

        rows.append({
            "raw_text": text + (" (paired)" if paired_from_next else ""),
            "item_guess": item_guess or "",
            "matched_item": matched_item,
            "match_score": round(score,2),
            "billed_price": price_val,
            "mrp": mrp,
            "unit": unit,
            "overcharge": overcharge,
            "status": status
        })

    return rows




# --------------------------
# PDF SAFE MULTI-CELL FIX
# --------------------------
def safe_multicell(pdf, text, h=7, max_len=60):
    """Prevents FPDF 'Not enough horizontal space' crash."""
    lines = text.split("\n")

    for line in lines:
        words = line.split(" ")
        current = ""

        for w in words:

            # If a very long word (no spaces) → break manually
            if len(w) > max_len:
                if current:
                    pdf.multi_cell(0, h, current)
                    current = ""
                for i in range(0, len(w), max_len):
                    pdf.multi_cell(0, h, w[i:i+max_len])
                continue

            # normal wrapping
            if len(current) + len(w) + 1 < max_len:
                current += w + " "
            else:
                pdf.multi_cell(0, h, current)
                current = w + " "

        if current:
            pdf.multi_cell(0, h, current)



# --------------------------
# Main UI
# --------------------------
if uploaded:
    bytes_data = uploaded.read()
    st.image(bytes_data, caption="Uploaded bill", use_column_width=True)

    with st.spinner("Running OCR and parsing..."):
        ocr_lines, img = extract_text_from_image(bytes_data)
        parsed_rows = parse_lines_horizontal(ocr_lines)

    if not parsed_rows:
        st.warning("No items detected.")
    else:
        df_res = pd.DataFrame(parsed_rows)
        st.subheader("Detected items (edit if OCR mismatch)")
        edited = st.data_editor(df_res, num_rows="dynamic", use_container_width=True)

        total_overcharge = edited['overcharge'].fillna(0).sum()
        total_overcharge = round(total_overcharge,2)

        st.metric("Total Overcharge (₹)", f"{total_overcharge}")

        st.subheader("Flagged items")
        flagged = edited[edited['status'].str.contains("Overcharged|Suspicious", na=False)]
        st.table(flagged[['raw_text','item_guess','matched_item','billed_price','mrp','overcharge','status']])

        # Export CSV
        csv = edited.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Report", data=csv, file_name="kirana_bill_report.csv")


        # -------- PDF GENERATION (FULLY FIXED) --------
        if st.button("Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()

            pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
            pdf.set_font("DejaVu", size=12)

            pdf.cell(0,10, "Kirana Bill Fraud Report", ln=True)
            pdf.cell(0,8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.ln(6)

            pdf.cell(0,8, f"Total Overcharge: ₹{total_overcharge}", ln=True)
            pdf.ln(6)
            pdf.cell(0,8, "Flagged items:", ln=True)

            for idx, row in flagged.iterrows():
                txt = (
                    f"- {row['raw_text']} | "
                    f"Billed: ₹{row['billed_price']} | "
                    f"MRP: {row['mrp']} | "
                    f"Overcharge: ₹{row['overcharge']} | "
                    f"Note: {row['status']}"
                )
                safe_multicell(pdf, txt)

            out = io.BytesIO()
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            out.write(pdf_bytes)
            out.seek(0)

            st.download_button(
                "Download PDF Report",
                data=out,
                file_name="kirana_fraud_report.pdf",
                mime="application/pdf"
            )
