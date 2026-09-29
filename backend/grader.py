import csv
import re
import os
from groq import Groq


client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)


def grade_student(answer_key_path, ocr_text):

    # -----------------------------
    # Read answer key
    # -----------------------------

    with open(answer_key_path, "r", encoding="utf-8") as file:
        answer_key = list(csv.DictReader(file))

    answer_key_text = ""

    for row in answer_key:
        answer_key_text += (
            f"Question {row['Question_Number']} "
            f"({row['Type']}): "
            f"{row['Correct_Answer']}\n"
        )

    # -----------------------------
    # Remove HTML tags only
    # -----------------------------

    ocr_text = re.sub(r"<[^>]+>", "", ocr_text)

    # -----------------------------
    # Prompt
    # -----------------------------

    prompt = f"""
You are an expert university exam evaluator.

You will receive:

1. The professor's answer key.
2. The complete OCR text of a student's handwritten answer sheet.

The OCR text may contain:
- spelling mistakes
- grammar mistakes
- OCR mistakes
- questions appearing out of order
- question text mixed with student answers
- missing punctuation
- bullet points
- page breaks
- formatting problems

Your job is to understand the student's answers based on their MEANING.

DO NOT rely on exact word-to-word matching.

========================
PROFESSOR ANSWER KEY
========================

{answer_key_text}

========================
STUDENT OCR TEXT
========================

{ocr_text}

========================
GRADING INSTRUCTIONS
========================

Questions 1-20 are MCQs.

For MCQs:
- Compare the student's selected option with the correct option.
- Correct answer = full marks.
- Incorrect answer = 0 marks.

Questions 21-35 are short-answer questions.

For short answers:
- Compare the student's answer with the professor's answer conceptually.
- Do NOT require the same wording.
- Ignore grammar and spelling mistakes when the intended meaning is clear.
- Ignore minor OCR errors when the intended meaning is understandable.
- Give full marks when the student demonstrates the required concept.
- Give partial marks when the student demonstrates only part of the required concept.
- Give 0 when the answer is incorrect, irrelevant, or missing.
- Do not penalize a student simply because they explain the concept differently from the professor.
- Do not invent information that the student did not write.

First identify which text belongs to each question.
The questions may appear out of order in the OCR text.

Then grade every question from 1 to 35.

Assume:
- Questions 1-20 = 1 mark each.
- Questions 21-35 = 3 marks each.

========================
OUTPUT FORMAT
========================

Return ONLY CSV data.

Do not use Markdown.
Do not use code fences.
Do not add explanations before or after the CSV.

The CSV must have exactly these columns:

Question_Number,Marks

Return exactly one row for every question from 1 to 35.

Rules:
- Questions 1-20: Marks must be either 0 or 1.
- Questions 21-35: Marks must be between 0 and 3.
- Use numeric values only for Marks.
- Do not write "1/1", "3/3", etc.
- Do not omit questions.
- Keep questions in numerical order.
"""

    # -----------------------------
    # Send to Qwen
    # -----------------------------

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        temperature=1,
        reasoning_effort="none",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content

    return content