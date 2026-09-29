import fitz
from PIL import Image
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor
import os


def ocr_pdf(pdf_path, output_folder="pdf_pages"):

    # -----------------------------
    # Create output folder
    # -----------------------------

    os.makedirs(output_folder, exist_ok=True)

    # -----------------------------
    # Load Surya OCR model
    # -----------------------------

    manager = SuryaInferenceManager()
    recognizer = RecognitionPredictor(manager)

    # -----------------------------
    # Open PDF
    # -----------------------------

    doc = fitz.open(pdf_path)

    complete_text = ""

    # -----------------------------
    # Process each PDF page
    # -----------------------------

    for i, page in enumerate(doc):

        # Convert PDF page to image
        pix = page.get_pixmap(dpi=300)

        image_path = os.path.join(
            output_folder,
            f"page_{i + 1}.png"
        )

        pix.save(image_path)

        # Open image
        image = Image.open(image_path)

        # Run OCR
        result = recognizer([image])

        # Get OCR result for this page
        page_result = result[0]

        page_text = ""

        # Sort blocks according to reading order
        blocks = sorted(
            page_result.blocks,
            key=lambda x: x.reading_order
        )

        for block in blocks:

            # Skip blocks marked as skipped
            if block.skipped:
                continue

            # Get text from HTML
            if block.html:
                page_text += block.html + "\n"

        # Add page separator
        complete_text += (
            f"\n\n{'=' * 80}\n"
            f"PAGE {i + 1}\n"
            f"{'=' * 80}\n\n"
        )

        complete_text += page_text

    doc.close()

    return complete_text