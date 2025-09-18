from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
app = Flask(__name__)
model = tf.keras.models.load_model("breed_recognition_model.h5")
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = Image.open(file).resize((224, 224))  # adjust to your input size
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            pred = model.predict(img_array)
            prediction = f"Predicted Breed: {np.argmax(pred)}"
    return render_template("index.html", prediction=prediction)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
