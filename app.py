import streamlit as st
import tempfile
import os
import re
import json

from backend.ocr import ocr_pdf
from backend.grader import grade_student


st.set_page_config(
    page_title="InkSwap",
    page_icon="📝",
    layout="wide"
)


# -----------------------------
# Check API key
# -----------------------------

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "🔑 GROQ_API_KEY is not set. Set it as an environment variable "
        "and restart the app."
    )
    st.stop()


# -----------------------------
# Saving results to disk
# -----------------------------

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results"
)


def results_file_path(exam_name):
    safe_name = re.sub(r"[^\w\-]+", "_", exam_name).strip("_") or "exam"
    return os.path.join(RESULTS_DIR, f"{safe_name}.json")


def load_results(exam_name):
    path = results_file_path(exam_name)

    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (OSError, ValueError):
        return []


def save_results(exam_name, results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    path = results_file_path(exam_name)
    temp_path = path + ".tmp"

    # Write to a temp file first so a crash can't corrupt the saved results
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    os.replace(temp_path, path)


# -----------------------------
# Marks validation helpers
# -----------------------------

TOTAL_QUESTIONS = 35
MCQ_COUNT = 20


def max_marks_for(question_number):
    # Q1-20 are MCQs (1 mark), Q21-35 are short answers (3 marks)
    return 1 if question_number <= MCQ_COUNT else 3


MARKS_LINE = re.compile(
    r"^(?:Q|Question)?\s*(\d+)\s*[,:\t]\s*(\d+(?:\.\d+)?|\.\d+)"
    r"\s*(?:/\s*\d+(?:\.\d+)?)?$",
    re.IGNORECASE
)


def parse_marks_csv(marks_csv):
    """
    Parse the LLM output safely.
    Returns (marks, issues):
      marks  -> {question_number (int): marks (float)} for valid rows only
      issues -> list of human-readable problems found
    """

    marks = {}
    issues = []

    for raw_line in marks_csv.splitlines():

        line = raw_line.replace('"', "").strip()

        # Skip blanks, code fences and the header row
        if not line or line.startswith("```"):
            continue

        if line.lower().startswith("question_number"):
            continue

        match = MARKS_LINE.match(line)

        if not match:
            issues.append(f"Ignored unreadable line: '{raw_line.strip()}'")
            continue

        question_number = int(match.group(1))
        value = float(match.group(2))

        if not 1 <= question_number <= TOTAL_QUESTIONS:
            issues.append(f"Ignored unknown question number: {question_number}")
            continue

        if question_number in marks:
            issues.append(f"Q{question_number} appeared more than once; kept the first value")
            continue

        maximum = max_marks_for(question_number)

        if value > maximum:
            issues.append(
                f"Q{question_number}: model gave {value:g}, "
                f"capped to the maximum of {maximum}"
            )
            value = maximum

        marks[question_number] = value

    return marks, issues


# -----------------------------
# Session state
# -----------------------------

if "evaluation_created" not in st.session_state:
    st.session_state.evaluation_created = False

if "exam_name" not in st.session_state:
    st.session_state.exam_name = ""

if "answer_key" not in st.session_state:
    st.session_state.answer_key = None

if "student_results" not in st.session_state:
    st.session_state.student_results = []

if "resumed_count" not in st.session_state:
    st.session_state.resumed_count = 0

# -----------------------------
# Header
# -----------------------------

st.title("📝 InkSwap - AI Answer Sheet Evaluator")
st.write("Automatically evaluate handwritten answer sheets using OCR and AI.")

st.divider()


# -----------------------------
# Create Evaluation
# -----------------------------

if not st.session_state.evaluation_created:

    st.header("Create New Evaluation")

    exam_name = st.text_input("Exam Name")

    answer_key = st.file_uploader(
        "Upload Answer Key",
        type=["csv"]
    )

    if st.button("Create Evaluation", type="primary"):

        if not exam_name:
            st.warning("Please enter the exam name.")

        elif answer_key is None:
            st.warning("Please upload the answer key.")

        else:
            st.session_state.evaluation_created = True
            st.session_state.exam_name = exam_name
            st.session_state.answer_key = answer_key

            # Bring back students already evaluated for this exam
            st.session_state.student_results = load_results(exam_name)
            st.session_state.resumed_count = len(
                st.session_state.student_results
            )

            st.rerun()


# -----------------------------
# Evaluation Created
# -----------------------------

else:

    st.header(f"📚 {st.session_state.exam_name}")

    st.success("Evaluation created successfully!")

    if st.session_state.resumed_count:
        st.info(
            f"Loaded {st.session_state.resumed_count} previously "
            f"evaluated student(s) for this exam."
        )

    st.write(
        f"Answer Key: **{st.session_state.answer_key.name}**"
    )

    st.divider()

    st.header("👨‍🎓 Students")

    st.subheader("Add Student")

    student_name = st.text_input("Student Name")

    roll_number = st.text_input("Roll Number")

    answer_sheet = st.file_uploader(
        "Upload Answer Sheet",
        type=["pdf"]
    )

    if st.button("Evaluate Student", type="primary"):

        if not student_name:
            st.warning("Please enter the student name.")

        elif not roll_number:
            st.warning("Please enter the roll number.")

        elif answer_sheet is None:
            st.warning("Please upload the answer sheet.")

        else:

            # Initialise paths so the finally block is always safe
            pdf_path = None
            answer_key_path = None

            with st.spinner("Processing answer sheet..."):

                # -----------------------------
                # Save uploaded PDF temporarily
                # -----------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(answer_sheet.getbuffer())
                    pdf_path = temp_file.name

                try:

                    # -----------------------------
                    # OCR
                    # -----------------------------

                    # -----------------------------
                    # OCR Progress Display
                    # -----------------------------

                    st.write("🔍 Running handwriting OCR...")

                    progress_box = st.empty()


                    def show_ocr_progress(message):
                        progress_box.info(f"🔄 {message}")


                    ocr_text = ocr_pdf(
                        pdf_path,
                        progress_callback=show_ocr_progress
                    )

                    progress_box.success("✅ OCR completed.")

                    # -----------------------------
                    # Save answer key temporarily
                    # -----------------------------

                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        delete=False,
                        suffix=".csv"
                    ) as key_file:

                        key_file.write(
                            st.session_state.answer_key.getbuffer()
                        )

                        answer_key_path = key_file.name

                    # -----------------------------
                    # AI Grading
                    # -----------------------------

                    st.write("🤖 Evaluating answers with AI...")

                    marks_csv = grade_student(
                        answer_key_path,
                        ocr_text
                    )
                    # -----------------------------
                    # Store Student Result
                    # -----------------------------

                    parsed_marks, issues = parse_marks_csv(marks_csv)

                    # Retry once if the model skipped any questions
                    missing = [
                        q for q in range(1, TOTAL_QUESTIONS + 1)
                        if q not in parsed_marks
                    ]

                    if missing:

                        st.write("⚠️ Some questions were missing. Retrying once...")

                        retry_csv = grade_student(
                            answer_key_path,
                            ocr_text
                        )

                        retry_marks, retry_issues = parse_marks_csv(retry_csv)

                        for q in missing:
                            if q in retry_marks:
                                parsed_marks[q] = retry_marks[q]

                        issues.extend(retry_issues)

                    # Anything still missing is set to 0 and flagged
                    still_missing = [
                        q for q in range(1, TOTAL_QUESTIONS + 1)
                        if q not in parsed_marks
                    ]

                    for q in still_missing:
                        parsed_marks[q] = 0.0

                    if still_missing:
                        issues.append(
                            "No marks returned for: "
                            + ", ".join(f"Q{q}" for q in still_missing)
                            + " (set to 0, please review manually)"
                        )

                    # Always exactly Q1..Q35, in order
                    student_marks = {
                        f"Q{q}": parsed_marks[q]
                        for q in range(1, TOTAL_QUESTIONS + 1)
                    }

                    # Validated CSV text for display
                    marks_csv = "Question_Number,Marks\n" + "\n".join(
                        f"{q},{parsed_marks[q]:g}"
                        for q in range(1, TOTAL_QUESTIONS + 1)
                    )


                    # Calculate total
                    total_marks = sum(student_marks.values())

                    # Maximum marks
                    maximum_marks = 20 + (15 * 3)

                    percentage = (total_marks / maximum_marks) * 100


                    # Add student information
                    student_result = {
                        "Student_Name": student_name,
                        "Roll_Number": roll_number,
                        **student_marks,
                        "Total_Marks": total_marks,
                        "Percentage": round(percentage, 2)
                    }

                    st.session_state.student_results.append(student_result)

                    try:
                        save_results(
                            st.session_state.exam_name,
                            st.session_state.student_results
                        )
                    except OSError as save_error:
                        st.warning(
                            f"Could not save results to disk: {save_error}"
                        )
                    st.success("Evaluation completed!")

                    # -----------------------------
                    # Display Result
                    # -----------------------------

                    st.subheader("📊 Result")

                    st.code(
                        marks_csv,
                        language="text"
                    )

                    if issues:
                        st.warning(
                            "Some problems were found in the AI output:\n\n"
                            + "\n".join(f"- {i}" for i in issues)
                        )

                except Exception as e:

                    st.error(f"Evaluation failed: {e}")

                finally:

                    if pdf_path and os.path.exists(pdf_path):
                        os.remove(pdf_path)

                    if answer_key_path and os.path.exists(answer_key_path):
                        os.remove(answer_key_path)

# -----------------------------
# Evaluated Students
# -----------------------------

if st.session_state.student_results:

    st.divider()

    st.subheader("👨‍🎓 Evaluated Students")

    # Create simple display table
    student_display = []

    for student in st.session_state.student_results:
        student_display.append({
            "Student Name": student["Student_Name"],
            "Roll Number": student["Roll_Number"],
            "Total Marks": f"{student['Total_Marks']:.0f} / 65",
            "Percentage": f"{student['Percentage']:.2f}%"
        })

    try:
        st.dataframe(
            student_display,
            width="stretch",
            hide_index=True
        )
    except Exception:
        # Older Streamlit versions don't support width="stretch"
        st.dataframe(
            student_display,
            use_container_width=True,
            hide_index=True
        )

    # -----------------------------
    # Generate CSV
    # -----------------------------

    import pandas as pd

    results_df = pd.DataFrame(
        st.session_state.student_results
    )

    # Make question columns appear in order
    question_columns = [
        f"Q{i}"
        for i in range(1, 36)
    ]

    final_columns = [
        "Student_Name",
        "Roll_Number",
        *question_columns,
        "Total_Marks",
        "Percentage"
    ]

    # Make sure missing question columns exist
    for column in question_columns:
        if column not in results_df.columns:
            results_df[column] = 0

    results_df = results_df[final_columns]

    # Generate CSV
    csv_data = results_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Generate Results CSV",
        data=csv_data,
        file_name=f"{st.session_state.exam_name}_results.csv",
        mime="text/csv"
    )