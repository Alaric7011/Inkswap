import fitz
from PIL import Image
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor
import os


def ocr_pdf(pdf_path, progress_callback=None):

    # -----------------------------
    # Create output folder
    # -----------------------------

    output_folder = "pdf_pages"
    os.makedirs(output_folder, exist_ok=True)

    # -----------------------------
    # Load Surya OCR model.
    # -----------------------------

    if progress_callback:
        progress_callback("Loading Surya OCR model...")

    manager = SuryaInferenceManager()
    recognizer = RecognitionPredictor(manager)

    # -----------------------------
    # Open PDF
    # -----------------------------

    doc = fitz.open(pdf_path)

    total_pages = len(doc)

    if progress_callback:
        progress_callback(
            f"PDF loaded — {total_pages} page(s) found."
        )

    complete_text = ""

    # -----------------------------
    # Process each PDF page
    # -----------------------------

    for i, page in enumerate(doc):

        if progress_callback:
            progress_callback(
                f"Running Surya OCR — Page {i + 1}/{total_pages}"
            )

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

        if progress_callback:
            progress_callback(
                f"Page {i + 1}/{total_pages} OCR completed."
            )

    doc.close()

    if progress_callback:
        progress_callback("All pages OCR completed.")

    return complete_text