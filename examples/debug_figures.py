import fitz
import sys
import os

pdf_path = "./downloaded_test_pdfs/1706.03762v7.pdf"

if not os.path.exists(pdf_path):
    print(f"Error: {pdf_path} not found.")
    sys.exit(1)

doc = fitz.open(pdf_path)
print(f"Successfully opened {pdf_path}. Total pages: {len(doc)}")

for page_num in range(len(doc)):
    page = doc[page_num]
    page_area = page.rect.width * page.rect.height
    images = page.get_image_info()
    if images:
        print(f"\n--- Page {page_num + 1} ---")
        print(f"Page Area: {page_area:.2f}")
        for idx, img in enumerate(images):
            x0, y0, x1, y1 = img["bbox"]
            img_area = (x1 - x0) * (y1 - y0)
            coverage = img_area / page_area
            xref = img.get("xref", 0)
            print(f"  Image {idx + 1}:")
            print(f"    BBox: ({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f})")
            print(f"    Area: {img_area:.2f} (Coverage: {coverage * 100:.2f}%)")
            print(f"    XRef: {xref}")
            
            # Check raw extraction
            try:
                img_data = doc.extract_image(xref)
                raw_len = len(img_data["image"]) if img_data else 0
                print(f"    Raw Image Bytes Length: {raw_len}")
            except Exception as e:
                print(f"    Raw Image Bytes Error: {e}")

doc.close()
