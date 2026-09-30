import fitz
from PIL import Image
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor
from surya.settings import settings


# -----------------------------
# Load the Surya model once per process
# -----------------------------
# Streamlit re-runs app.py on every interaction, but imported modules stay
# loaded, so this module-level variable survives between students and reruns.

_recognizer = None


def get_recognizer(progress_callback=None):

    global _recognizer

    if _recognizer is None:

        if progress_callback:
            progress_callback("Loading Surya OCR model (first time only)...")

        manager = SuryaInferenceManager()
        _recognizer = RecognitionPredictor(manager)

    return _recognizer


def ocr_pdf(pdf_path, progress_callback=None):

    recognizer = get_recognizer(progress_callback)

    # PDF points are 1/72 inch, so this renders each page at Surya's
    # high-res DPI setting instead of a fixed 300
    scale_factor = settings.IMAGE_DPI_HIGHRES / 72
    matrix = fitz.Matrix(scale_factor, scale_factor)

    # -----------------------------
    # Open PDF
    # -----------------------------

    doc = fitz.open(pdf_path)

    try:

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

            # Render the page straight to memory (no student images
            # are written to disk)
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            image = Image.frombytes(
                "RGB",
                (pix.width, pix.height),
                pix.samples
            )

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

    finally:
        doc.close()

    if progress_callback:
        progress_callback("All pages OCR completed.")

    return complete_text
