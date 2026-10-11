import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = np.full((120, 700, 3), 255, np.uint8)
cv2.putText(img, "Internship Evaluation 2026", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
print("OCR:", pytesseract.image_to_string(img).strip())

doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 100), "PDF render test")
pix = page.get_pixmap(dpi=300)
print("PDF -> image:", pix.width, "x", pix.height)

print("OpenCV:", cv2.__version__)
print("OK")