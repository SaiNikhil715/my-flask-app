from flask import Flask, request, render_template_string
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ---------------------------
# Load model
# ---------------------------
MODEL_PATH = "breed_recognition_model.h5"

try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    print("Error loading model:", e)
    model = None

# ---------------------------
# Flask setup
# ---------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Replace these with your actual class labels
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
        # Check if file is uploaded
        if "file" not in request.files:
            return render_template_string(HTML_PAGE, prediction="No file uploaded")
        file = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))  # adjust size for your model
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0  # normalize if your model was trained with [0,1]

        # Predict
        pred = model.predict(x)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_breed = LABELS[pred_class]

        return render_template_string(HTML_PAGE, prediction=f"Predicted Breed: {pred_breed}")
    except Exception as e:
        return f"Error: {e}"

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
