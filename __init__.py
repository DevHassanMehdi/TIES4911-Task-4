import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask app and API
app = Flask(__name__)
api = Api(app)

# Load trained Keras model
MODEL_PATH = "fine_tuned_vgg16.h5"  # Update with actual model path
model = load_model(MODEL_PATH)

# Load class indices (Modify this part if needed)
class_indices = {0: 'cat', 1: 'dog'}  # Update based on your dataset


# Define image transformation
def transform_image(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	img_array = image.img_to_array(img)
	img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
	return img_array


# Define the API resource
class Predict(Resource):
	def post(self):
		if 'file' not in request.files:
			return jsonify({"error": "No file part"})
		file = request.files['file']
		if file.filename == '':
			return jsonify({"error": "No selected file"})
		
		filename = secure_filename(file.filename)
		filepath = os.path.join("temp", filename)
		os.makedirs("temp", exist_ok=True)
		file.save(filepath)
		
		image_tensor = transform_image(filepath)
		preds = model.predict(image_tensor)
		class_idx = np.argmax(preds[0])
		class_label = class_indices[class_idx]
		confidence = float(preds[0][class_idx])
		
		os.remove(filepath)  # Clean up the temp file
		return jsonify({"prediction": class_label, "confidence": confidence})


# Add the resource to API
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=5000)
