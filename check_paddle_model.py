import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    lang="it",
)

print("=" * 60)
print("PaddleOCR Model Info")
print("=" * 60)

model_dir = getattr(ocr, "model_dir", None)
print(f"model_dir: {model_dir}")

if model_dir and os.path.isdir(model_dir):
    print(f"\nContents of {model_dir}:")
    for root, dirs, files in os.walk(model_dir):
        level = root.replace(model_dir, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 2 * (level + 1)
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f))
            print(f"{sub_indent}{f} ({size / 1024 / 1024:.1f} MB)")
else:
    print(f"model_dir does not exist or is not set")

print("\n" + "=" * 60)
print("PaddleOCR version (from package):")
try:
    import paddleocr
    print(f"  paddleocr package: {paddleocr.__version__ if hasattr(paddleocr, '__version__') else 'unknown'}")
except Exception as e:
    print(f"  Could not get paddleocr version: {e}")

try:
    import paddle
    print(f"  paddle version: {paddle.__version__}")
except Exception:
    pass

print("=" * 60)