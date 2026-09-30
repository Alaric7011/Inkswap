import streamlit as st
import os
import re
import json

import pandas as pd

from backend.pipeline import evaluate_one_student, parse_filename
from backend.grader import (
    parse_answer_key,
    total_marks,
    describe_key,
    AnswerKeyError,
)


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


def split_compatible_results(saved_rows, answer_key):
    """
    Keep only saved rows scored on the SAME scale as the current answer key
    (same Max_Marks and same question columns). Older rows, e.g. from the
    previous 65-mark scheme, are returned separately so they are never
    silently mixed into a class CSV.
    """

    maximum = total_marks(answer_key)
    question_columns = [f"Q{r['number']}" for r in answer_key]

    compatible = []
    incompatible = []

    for row in saved_rows:
        if (
            isinstance(row, dict)
            and row.get("Max_Marks") == maximum
            and all(column in row for column in question_columns)
        ):
            compatible.append(row)
        else:
            incompatible.append(row)

    return compatible, incompatible


def normalize_roll(value):
    return str(value or "").strip().casefold()


def clean_cell(value):
    # Editor cells can come back as None / NaN when cleared
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def upsert_result(results, new_row):
    """Replace the row with the same roll number, or append a new one."""

    key = normalize_roll(new_row["Roll_Number"])

    for index, row in enumerate(results):
        if normalize_roll(row.get("Roll_Number")) == key:
            results[index] = new_row
            return

    results.append(new_row)


def show_table(container, data, editable=False, **kwargs):
    """Full-width table that works on old and new Streamlit versions."""

    function = container.data_editor if editable else container.dataframe

    try:
        return function(data, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        return function(data, use_container_width=True, hide_index=True, **kwargs)


# -----------------------------
# Session state
# -----------------------------

if "evaluation_created" not in st.session_state:
    st.session_state.evaluation_created = False

if "exam_name" not in st.session_state:
    st.session_state.exam_name = ""

if "answer_key_name" not in st.session_state:
    st.session_state.answer_key_name = ""

if "answer_key" not in st.session_state:
    # Parsed answer key: list of {number, type, answer, max_marks}
    st.session_state.answer_key = None

if "student_results" not in st.session_state:
    st.session_state.student_results = []

if "resumed_count" not in st.session_state:
    st.session_state.resumed_count = 0

if "ignored_count" not in st.session_state:
    st.session_state.ignored_count = 0

if "batch_signature" not in st.session_state:
    st.session_state.batch_signature = None

if "batch_version" not in st.session_state:
    st.session_state.batch_version = 0

if "batch_df" not in st.session_state:
    st.session_state.batch_df = None

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

    answer_key_file = st.file_uploader(
        "Upload Answer Key",
        type=["csv"]
    )

    st.caption(
        "CSV columns: Question_Number, Type (MCQ or Short_Answer), "
        "Correct_Answer. MCQ = 1 mark, Short_Answer = 2 marks."
    )

    if st.button("Create Evaluation", type="primary"):

        if not exam_name:
            st.warning("Please enter the exam name.")

        elif answer_key_file is None:
            st.warning("Please upload the answer key.")

        else:
            # Parse and validate the key once, up front
            try:
                parsed_key = parse_answer_key(answer_key_file.getvalue())

            except AnswerKeyError as key_error:
                st.error(f"Answer key problem: {key_error}")
                st.stop()

            st.session_state.evaluation_created = True
            st.session_state.exam_name = exam_name
            st.session_state.answer_key_name = answer_key_file.name
            st.session_state.answer_key = parsed_key

            # Bring back students already evaluated for this exam,
            # but only those scored on the same marks scale
            saved_rows = load_results(exam_name)

            compatible, incompatible = split_compatible_results(
                saved_rows,
                parsed_key
            )

            st.session_state.student_results = compatible
            st.session_state.resumed_count = len(compatible)
            st.session_state.ignored_count = len(incompatible)

            st.rerun()


# -----------------------------
# Evaluation Created
# -----------------------------

else:

    answer_key = st.session_state.answer_key
    maximum_marks = total_marks(answer_key)
    question_numbers = [r["number"] for r in answer_key]

    st.header(f"📚 {st.session_state.exam_name}")

    st.success("Evaluation created successfully!")

    if st.session_state.resumed_count:
        st.info(
            f"Loaded {st.session_state.resumed_count} previously "
            f"evaluated student(s) for this exam."
        )

    if st.session_state.ignored_count:
        st.warning(
            f"{st.session_state.ignored_count} saved result(s) for this "
            f"exam were scored on a different marks scale and were NOT "
            f"loaded. They will be replaced the next time results are saved."
        )

    st.write(
        f"Answer Key: **{st.session_state.answer_key_name}**"
    )

    st.write(f"Paper: **{describe_key(answer_key)}**")

    st.divider()

    st.header("👨‍🎓 Students")

    st.subheader("Upload Answer Sheets")

    st.caption(
        "Upload one or many PDFs. Names and roll numbers are guessed from "
        "the file names (e.g. 101_Ali_Khan.pdf); fix any of them in the "
        "table before starting."
    )

    files = st.file_uploader(
        "Upload Answer Sheets (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if not files:
        st.session_state.batch_signature = None

    else:

        # Rebuild the editable table only when the set of files changes,
        # so the teacher's edits are not wiped on every rerun
        signature = tuple((f.name, f.size) for f in files)

        if st.session_state.batch_signature != signature:

            rows = []

            for f in files:
                guessed_name, guessed_roll = parse_filename(f.name)
                rows.append({
                    "File": f.name,
                    "Student_Name": guessed_name,
                    "Roll_Number": guessed_roll,
                })

            st.session_state.batch_df = pd.DataFrame(rows)
            st.session_state.batch_signature = signature
            st.session_state.batch_version += 1

        edited_df = show_table(
            st,
            st.session_state.batch_df,
            editable=True,
            key=f"batch_editor_{st.session_state.batch_version}",
            disabled=["File"]
        )

        reevaluate = st.checkbox(
            "Re-evaluate students who already have results "
            "(replaces their old row)",
            value=False
        )

        st.caption(
            "Don't click around while a batch is running: any interaction "
            "restarts the page and stops the batch. Finished students are "
            "saved after each one, so just press the button again to "
            "continue; already-evaluated students are skipped."
        )

        if st.button("Evaluate All", type="primary"):

            # -----------------------------
            # Validate the table before starting
            # -----------------------------

            entries = []
            problems = []
            rolls_in_batch = {}

            for f, row in zip(files, edited_df.to_dict("records")):

                name = clean_cell(row.get("Student_Name"))
                roll = clean_cell(row.get("Roll_Number"))

                if not name or not roll:
                    problems.append(
                        f"{f.name}: student name and roll number are both required."
                    )
                    continue

                key = normalize_roll(roll)

                if key in rolls_in_batch:
                    problems.append(
                        f"Roll number '{roll}' is used by both "
                        f"{rolls_in_batch[key]} and {f.name}."
                    )
                    continue

                rolls_in_batch[key] = f.name
                entries.append((f, name, roll))

            if problems:

                st.error(
                    "Fix these before starting:\n\n"
                    + "\n".join(f"- {p}" for p in problems)
                )

            else:

                existing_rolls = {
                    normalize_roll(r.get("Roll_Number"))
                    for r in st.session_state.student_results
                }

                total_files = len(entries)

                status_rows = [
                    {
                        "File": f.name,
                        "Student": name,
                        "Roll": roll,
                        "Status": "Queued",
                        "Notes": "",
                    }
                    for f, name, roll in entries
                ]

                overall_bar = st.progress(0.0, text="Starting batch...")
                current_line = st.empty()
                status_box = st.empty()

                show_table(status_box, pd.DataFrame(status_rows))

                counts = {"done": 0, "review": 0, "failed": 0, "skipped": 0}

                for index, (f, name, roll) in enumerate(entries):

                    label = f"{index + 1}/{total_files}: {name} ({roll})"

                    # Resume support: skip students already evaluated
                    if (
                        normalize_roll(roll) in existing_rolls
                        and not reevaluate
                    ):
                        status_rows[index]["Status"] = "Skipped"
                        status_rows[index]["Notes"] = "Already evaluated"
                        counts["skipped"] += 1
                        show_table(status_box, pd.DataFrame(status_rows))
                        overall_bar.progress(
                            (index + 1) / total_files,
                            text=f"Finished {index + 1} of {total_files}"
                        )
                        continue

                    status_rows[index]["Status"] = "Running"
                    show_table(status_box, pd.DataFrame(status_rows))
                    overall_bar.progress(
                        index / total_files,
                        text=f"Evaluating {label}"
                    )

                    def show_progress(message, label=label):
                        current_line.info(f"🔄 {label} — {message}")

                    try:

                        outcome = evaluate_one_student(
                            f.getvalue(),
                            answer_key,
                            progress_callback=show_progress
                        )

                        student_marks = {
                            f"Q{q}": marks
                            for q, marks in outcome["marks"].items()
                        }

                        total = sum(student_marks.values())
                        issues = outcome["issues"]

                        student_result = {
                            "Student_Name": name,
                            "Roll_Number": roll,
                            **student_marks,
                            "Total_Marks": total,
                            "Max_Marks": maximum_marks,
                            "Percentage": round(
                                (total / maximum_marks) * 100, 2
                            ),
                            "Needs_Review": "Yes" if issues else "",
                            "Review_Notes": " | ".join(issues),
                        }

                        upsert_result(
                            st.session_state.student_results,
                            student_result
                        )

                        # Save after EVERY student so a crash loses at most one
                        try:
                            save_results(
                                st.session_state.exam_name,
                                st.session_state.student_results
                            )
                        except OSError as save_error:
                            issues = issues + [
                                f"Could not save to disk: {save_error}"
                            ]

                        if issues:
                            status_rows[index]["Status"] = "Done — review"
                            status_rows[index]["Notes"] = issues[0] + (
                                f" (+{len(issues) - 1} more)"
                                if len(issues) > 1 else ""
                            )
                            counts["review"] += 1
                        else:
                            status_rows[index]["Status"] = "Done"
                            counts["done"] += 1

                    except Exception as error:

                        status_rows[index]["Status"] = "Failed"
                        status_rows[index]["Notes"] = str(error)
                        counts["failed"] += 1

                    show_table(status_box, pd.DataFrame(status_rows))
                    overall_bar.progress(
                        (index + 1) / total_files,
                        text=f"Finished {index + 1} of {total_files}"
                    )

                current_line.empty()

                summary = (
                    f"Batch finished: {counts['done']} done, "
                    f"{counts['review']} need review, "
                    f"{counts['failed']} failed, "
                    f"{counts['skipped']} skipped."
                )

                if counts["failed"]:
                    st.warning(
                        summary + " Press Evaluate All again to retry the "
                        "failed ones; finished students are skipped."
                    )
                elif counts["review"]:
                    st.warning(summary)
                else:
                    st.success(summary)

# -----------------------------
# Evaluated Students
# -----------------------------

if st.session_state.evaluation_created and st.session_state.student_results:

    answer_key = st.session_state.answer_key
    maximum_marks = total_marks(answer_key)

    st.divider()

    st.subheader("👨‍🎓 Evaluated Students")

    # Create simple display table
    student_display = []

    for student in st.session_state.student_results:
        student_display.append({
            "Student Name": student["Student_Name"],
            "Roll Number": student["Roll_Number"],
            "Total Marks": f"{student['Total_Marks']:g} / {maximum_marks:g}",
            "Percentage": f"{student['Percentage']:.2f}%",
            "Needs Review": student.get("Needs_Review", "")
        })

    show_table(st, student_display)

    flagged = [
        s for s in st.session_state.student_results
        if s.get("Needs_Review")
    ]

    if flagged:
        with st.expander(f"⚠️ Review notes ({len(flagged)} student(s))"):
            for s in flagged:
                st.markdown(
                    f"**{s['Student_Name']} ({s['Roll_Number']})**"
                )
                for note in s.get("Review_Notes", "").split(" | "):
                    st.write(f"- {note}")

    # -----------------------------
    # Generate CSV
    # -----------------------------

    results_df = pd.DataFrame(
        st.session_state.student_results
    )

    # Question columns come from the answer key, in order
    question_columns = [
        f"Q{r['number']}"
        for r in answer_key
    ]

    final_columns = [
        "Student_Name",
        "Roll_Number",
        *question_columns,
        "Total_Marks",
        "Max_Marks",
        "Percentage",
        "Needs_Review",
        "Review_Notes"
    ]

    # Older saved rows may lack the review columns
    results_df = results_df.reindex(columns=final_columns, fill_value="")

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
