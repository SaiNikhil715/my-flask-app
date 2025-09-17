from flask import Flask, request, render_template_string
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
import gdown

# ---------------------------
# Paths and Google Drive
# ---------------------------
MODEL_PATH = "breed_recognition_model.h5"
FILE_ID = "https://drive.google.com/file/d/1GPrNFPJ9txqsID_jlA0UmZaQBOcf68xU/view?usp=sharing"  # Replace with your Drive model ID

# ---------------------------
# Function to create dummy model
# ---------------------------
def create_dummy_model():
    dummy = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Flatten(),
        layers.Dense(5, activation='softmax')
    ])
    dummy.compile(optimizer='adam', loss='categorical_crossentropy')
    dummy.save(MODEL_PATH)
    print("Dummy model created.")

# ---------------------------
# Download or load model
# ---------------------------
if not os.path.exists(MODEL_PATH):
    try:
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        print("Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
    except Exception as e:
        print("Download failed, creating dummy model instead.", e)
        create_dummy_model()

# Load model
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print("Failed to load model, using dummy model.", e)
    create_dummy_model()
    model = keras.models.load_model(MODEL_PATH)

# ---------------------------
# Flask setup
# ---------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

LABELS = ["Beagle", "Pug", "Bulldog", "Golden Retriever", "Labrador"]

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Breed Recognition Prototype</title></head>
<body>
<h2>✅ Breed Recognition Prototype</h2>
<form method="POST" action="/predict" enctype="multipart/form-data">
<input type="file" name="file" accept="image/*" required>
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
        return "Model not loaded"

    try:
        if "file" not in request.files:
            return render_template_string(HTML_PAGE, prediction="No file uploaded")
        file = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Preprocess image
        img = load_img(filepath, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0

        # Predict
        pred = model.predict(x)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_breed = LABELS[pred_class]

        return render_template_string(HTML_PAGE, prediction=f"Predicted Breed: {pred_breed}")
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
