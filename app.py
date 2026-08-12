import os
import tempfile
from flask import Flask, request, render_template_string
from texteller import load_model, load_tokenizer, img2latex

app = Flask(__name__)

model = load_model(use_onnx=False)
tokenizer = load_tokenizer()

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TexTeller</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 20px; }
    input, button { font-size: 16px; margin: 8px 0; }
    pre { white-space: pre-wrap; background: #f4f4f4; padding: 16px; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>TexTeller</h1>
  <form method="post" action="/predict" enctype="multipart/form-data">
    <input type="file" name="img" accept="image/*" required>
    <br>
    <button type="submit">Recognize formula</button>
  </form>
  {% if latex is not none %}
    <h2>LaTeX</h2>
    <pre>{{ latex }}</pre>
  {% endif %}
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(PAGE, latex=None)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    file = request.files.get("img")
    if not file or not file.filename:
        return "No image uploaded", 400

    suffix = os.path.splitext(file.filename)[1] or ".png"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        latex = img2latex(model, tokenizer, [tmp_path])[0]

        if "text/html" in request.headers.get("Accept", ""):
            return render_template_string(PAGE, latex=latex)

        return str(latex)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
