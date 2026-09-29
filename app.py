import streamlit as st
import tempfile
import os

from backend.ocr import ocr_pdf
from backend.grader import grade_student


st.set_page_config(
    page_title="AI Answer Sheet Evaluator",
    page_icon="📝",
    layout="wide"
)


# -----------------------------
# Session state
# -----------------------------

if "evaluation_created" not in st.session_state:
    st.session_state.evaluation_created = False

if "exam_name" not in st.session_state:
    st.session_state.exam_name = ""

if "answer_key" not in st.session_state:
    st.session_state.answer_key = None


# -----------------------------
# Header
# -----------------------------

st.title("📝 AI Answer Sheet Evaluator")
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

            st.rerun()


# -----------------------------
# Evaluation Created
# -----------------------------

else:

    st.header(f"📚 {st.session_state.exam_name}")

    st.success("Evaluation created successfully!")

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

                    st.write("🔍 Running handwriting OCR...")

                    ocr_text = ocr_pdf(pdf_path)

                    st.success("OCR completed.")

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

                    st.success("Evaluation completed!")

                    # -----------------------------
                    # Display Result
                    # -----------------------------

                    st.subheader("📊 Result")

                    st.code(
                        marks_csv,
                        language="text"
                    )

                finally:

                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)

                    if os.path.exists(answer_key_path):
                        os.remove(answer_key_path)