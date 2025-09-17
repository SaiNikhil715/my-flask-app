from flask import Flask, request, render_template_string
from tensorflow import keras
import numpy as np
import os
import gdown

# ----------------------------
# Google Drive model settings
# ----------------------------
FILE_ID = "1GPrNFPJ9txqsID_jlA0UmZaQBOcf68xU"
MODEL_PATH = "breed_recognition_model.h5"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"  # Correct f-string
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model safely
try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    print("Error loading model:", e)
    model = None  # Prevent crash if model fails

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Flask ML Prototype</title></head>
<body>
<h2>✅ ML Prototype</h2>
<form method="POST" action="/predict">
<input type="text" name="input_data" placeholder="comma separated numbers" size="40">
<button type="submit">Predict</button>
</form>
{% if prediction is not none %}
<h3>Prediction: {{ prediction }}</h3>
{% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not loaded. Check server logs."
    try:
        # Read and convert input
        data = request.form.get("input_data", "")
        if not data:
            return render_template_string(HTML_PAGE, prediction="No input provided")

        numbers = np.array([float(x.strip()) for x in data.split(",")]).reshape(1, -1)

        # Prediction
        prediction = model.predict(numbers)
        prediction = prediction.tolist()  # Convert to Python list

        return render_template_string(HTML_PAGE, prediction=prediction)
    except Exception as e:
        return f"Error processing input: {e}"

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
from flask import Flask, request, render_template_string
from tensorflow import keras
import numpy as np
import os
import gdown

# ----------------------------
# Google Drive model settings
# ----------------------------
FILE_ID = "1GPrNFPJ9txqsID_jlA0UmZaQBOcf68xU"
MODEL_PATH = "breed_recognition_model.h5"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"  # Correct f-string
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model safely
try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    print("Error loading model:", e)
    model = None  # Prevent crash if model fails

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Flask ML Prototype</title></head>
<body>
<h2>✅ ML Prototype</h2>
<form method="POST" action="/predict">
<input type="text" name="input_data" placeholder="comma separated numbers" size="40">
<button type="submit">Predict</button>
</form>
{% if prediction is not none %}
<h3>Prediction: {{ prediction }}</h3>
{% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not loaded. Check server logs."
    try:
        # Read and convert input
        data = request.form.get("input_data", "")
        if not data:
            return render_template_string(HTML_PAGE, prediction="No input provided")

        numbers = np.array([float(x.strip()) for x in data.split(",")]).reshape(1, -1)

        # Prediction
        prediction = model.predict(numbers)
        prediction = prediction.tolist()  # Convert to Python list

        return render_template_string(HTML_PAGE, prediction=prediction)
    except Exception as e:
        return f"Error processing input: {e}"

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
