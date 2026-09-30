import os
import re
import tempfile
import time

from backend.ocr import ocr_pdf
from backend.grader import grade_student, parse_marks_csv


# Groq errors worth retrying (rate limit / network hiccups).
# Anything else (bad key, bad request) fails immediately.
RETRYABLE_ERRORS = {
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
}

BACKOFF_SECONDS = (2, 5, 10)


def parse_filename(filename):
    """
    Guess (student_name, roll_number) from a file name.
      101_Ali_Khan.pdf   -> ("Ali Khan", "101")
      Ali-Khan-101.pdf   -> ("Ali Khan", "101")
      Student_3.pdf      -> ("Student", "3")
      scan.pdf           -> ("scan", "")
    The roll number is the first token that contains a digit.
    """

    stem = os.path.splitext(os.path.basename(filename))[0]

    tokens = [t for t in re.split(r"[_\-\s]+", stem) if t]

    roll = ""
    name_tokens = []

    for token in tokens:
        if not roll and re.search(r"\d", token):
            roll = token
        else:
            name_tokens.append(token)

    return " ".join(name_tokens), roll


def _grade_with_backoff(answer_key, ocr_text):

    for attempt in range(len(BACKOFF_SECONDS) + 1):

        try:
            return grade_student(answer_key, ocr_text)

        except Exception as error:

            retryable = type(error).__name__ in RETRYABLE_ERRORS
            last_attempt = attempt == len(BACKOFF_SECONDS)

            if not retryable or last_attempt:
                raise

            time.sleep(BACKOFF_SECONDS[attempt])


def evaluate_one_student(pdf_bytes, answer_key, progress_callback=None):
    """
    Full pipeline for ONE student: OCR -> AI grading -> validation.

    Returns {"marks": {question_number: int}, "issues": [str]}.
    Raises an exception if OCR or grading fails outright; the caller decides
    what to do (the batch loop records the failure and moves on).
    """

    question_numbers = [r["number"] for r in answer_key]

    # -----------------------------
    # OCR
    # -----------------------------

    pdf_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(pdf_bytes)
            pdf_path = temp_file.name

        ocr_text = ocr_pdf(pdf_path, progress_callback=progress_callback)

    finally:

        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)

    # Page separators alone don't count as text
    body = re.sub(r"=+|PAGE \d+", "", ocr_text).strip()

    if not body:
        raise ValueError("OCR found no text in this PDF (blank or unreadable scan).")

    # -----------------------------
    # AI grading
    # -----------------------------

    if progress_callback:
        progress_callback("Grading answers with AI...")

    marks_csv = _grade_with_backoff(answer_key, ocr_text)

    parsed_marks, issues = parse_marks_csv(marks_csv, answer_key)

    # Retry once if the model skipped any questions
    missing = [q for q in question_numbers if q not in parsed_marks]

    if missing:

        if progress_callback:
            progress_callback("Some questions were missing, retrying once...")

        retry_csv = _grade_with_backoff(answer_key, ocr_text)

        retry_marks, retry_issues = parse_marks_csv(retry_csv, answer_key)

        for q in missing:
            if q in retry_marks:
                parsed_marks[q] = retry_marks[q]

        issues.extend(retry_issues)

    # Anything still missing is set to 0 and flagged
    still_missing = [q for q in question_numbers if q not in parsed_marks]

    for q in still_missing:
        parsed_marks[q] = 0

    if still_missing:
        issues.append(
            "No marks returned for: "
            + ", ".join(f"Q{q}" for q in still_missing)
            + " (set to 0, please review manually)"
        )

    return {
        "marks": {q: parsed_marks[q] for q in question_numbers},
        "issues": issues,
    }
