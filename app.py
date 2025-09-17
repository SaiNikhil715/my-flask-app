import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ======================
# Load trained model
# ======================
MODEL_PATH = "breed_recognition_model.h5"  # trained model file
model = load_model(MODEL_PATH)

# You must ensure that the class labels are in the same order as training
# Replace these with your actual breeds from dataset
class_labels = ["Sahiwal", "Gir", "Murrah", "Jaffarabadi"]

# Flask app
app = Flask(__name__)

# Home page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No image selected"

    if file:
        # Save uploaded image temporarily
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        breed = class_labels[class_index]

        return render_template("index.html", breed=breed, img_path=filepath)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
