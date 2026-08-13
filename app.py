import os
import tempfile
import torch

from flask import Flask, request, render_template_string
from texteller import load_model, load_tokenizer, img2latex
from openai import OpenAI


app = Flask(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

model = load_model(use_onnx=True)
tokenizer = load_tokenizer()


# --------------------------------------------------
# CONTROL TEST
# For now we use one fixed problem.
# Later we can easily add Problem 2, Problem 3, etc.
# --------------------------------------------------

PROBLEM_TEXT = "Evaluate the indefinite integral:"
PROBLEM_LATEX = r"\int \sin x\,dx"


PAGE = """
<!doctype html>
<html lang="en">

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]
      }
    };
  </script>

  <script
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
  </script>

  <title>Math Control Test</title>

  <style>
    body {
      font-family: Arial, sans-serif;
      max-width: 760px;
      margin: 40px auto;
      padding: 0 20px;
    }

    input, button {
      font-size: 16px;
      margin: 8px 0;
    }

    pre {
      white-space: pre-wrap;
      background: #f4f4f4;
      padding: 16px;
      border-radius: 8px;
    }

    .problem {
      font-size: 22px;
      margin: 20px 0;
      padding: 15px;
      background: #f7f7f7;
      border-radius: 8px;
    }
  </style>
</head>


<body>

  <h1>Math Control Test</h1>

 <h2>Problem 1</h2>
<p>Write your solution and upload it below.</p>


  <form method="post"
        action="/predict"
        enctype="multipart/form-data">

    <label>
      Student name or ID:
    </label>
    <br>

    <input
      type="text"
      name="student"
      required
      value="{{ student or '' }}"
    >

    <br><br>

    <label>
      Upload your handwritten solution:
    </label>
    <br>

    <input
      type="file"
      name="img"
      accept="image/*"
      required
    >

    <br>

    <button type="submit">
      Submit solution
    </button>

  </form>


  {% if latex is not none %}

    <hr>

    <h2>Student</h2>
    <p>{{ student }}</p>


    <h2>Recognized solution</h2>

    <div style="font-size: 24px; margin: 20px 0;">
      \\({{ latex | safe }}\\)
    </div>


    <h2>LaTeX</h2>

    <pre>{{ latex }}</pre>


    {% if feedback %}

      <h2>AI check</h2>

      <pre>{{ feedback }}</pre>

    {% endif %}


  {% endif %}

</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(
        PAGE,
        latex=None,
        feedback=None,
        student=None,
        problem_text=PROBLEM_TEXT,
        problem_latex=PROBLEM_LATEX
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict():

    student = request.form.get("student", "").strip()

    if not student:
        return "Student name or ID is required", 400


    file = request.files.get("img")

    if not file or not file.filename:
        return "No image uploaded", 400


    suffix = os.path.splitext(file.filename)[1] or ".png"

    tmp_path = None


    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp_path = tmp.name
            file.save(tmp_path)


        # ------------------------------------------
        # OCR: handwriting -> LaTeX
        # ------------------------------------------

        with torch.inference_mode():

            latex = img2latex(
                model,
                tokenizer,
                [tmp_path]
            )[0]


        # ------------------------------------------
        # AI grading
        # ------------------------------------------

        try:

            response = client.responses.create(

                model="gpt-5.6-luna",

                input=f"""
You are grading a student's short mathematics control-test solution.

Problem:
{PROBLEM_TEXT}

Problem in LaTeX:
{PROBLEM_LATEX}

Recognized student solution:
{latex}

Important instructions:

- Grade the student's solution to the stated problem.
- Do not assume missing work was performed.
- If the OCR result appears incomplete or suspicious, say so.
- If you are not confident because handwriting may have been recognized incorrectly,
  do not give a definitive failing grade. State that teacher review is required.

Return exactly in this format:

Score: X/10
Result: Correct / Partially correct / Incorrect / Teacher review required
Comment: short explanation
"""

            )

            feedback = response.output_text


        except Exception as e:

            feedback = (
                "AI check unavailable. "
                "The handwritten solution was recognized, "
                "but automatic grading could not be completed."
            )

            print("OpenAI error:", repr(e))


        return render_template_string(

            PAGE,

            latex=latex,
            feedback=feedback,
            student=student,
            problem_text=PROBLEM_TEXT,
            problem_latex=PROBLEM_LATEX

        )


    finally:

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "10000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
