form flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    image_array = image / 255.0
    return image_array
    #TODO add implemenation for more types of photos and dimensions

def predict(image):
    image_array = preprocess_image(image)
    prediction = model.predict(np.expand_dims(image_array, axis=0)
    #TODO: return the predicted class based on model of choice

@app.route('/api', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    image_file = request.files['image']
    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    filename = secure_filename(image_file.filename)
    image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    image_path = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    image_file.save(image_path)
    predicted_class = predict_image(image_path)
    if os.path.exists(image_path):
        os.remove(image_path)
    return jsonify({'prediction': predicted_class})
    if __name__ == '__main__':
        app.run(debug=True)
