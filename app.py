from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/saved_model')

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict_image(image_path):
    img = load_img(image_path, target_size=(100, 150))
    x = img_to_array(img)

    if x.shape[-1] != 3:
        x = np.stack((x,) * 3, axis=-1)

    x = x / 255.0  # Normalisasi gambar
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)
    predicted_class_index = np.argmax(classes)
    probability = np.max(classes) * 100

    if predicted_class_index == 0:
        return "Paper", probability
    elif predicted_class_index == 1:
        return "Rock", probability
    else:
        return "Scissors", probability

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            prediction, probability = predict_image(file_path)
            image_path = file.filename
            return render_template('index.html', prediction=prediction, probability=probability, image_path=image_path)
    return render_template('index.html', prediction=None, probability=None, image_path=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
