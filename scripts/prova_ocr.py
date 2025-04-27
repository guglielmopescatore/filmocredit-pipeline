import easyocr
import numpy as np
from PIL import Image

reader = easyocr.Reader(['it', 'en'], gpu=True)  # usa gpu=False se hai problemi

image = Image.open("frame_test_titolo_iniziale.jpg")
image_np = np.array(image)

results = reader.readtext(image_np)

if results:
    print("✅ Testo rilevato:")
    for bbox, text, conf in results:
        print(f"- '{text}' (confidenza: {conf:.2f})")
else:
    print("❌ Nessun testo rilevato.")
