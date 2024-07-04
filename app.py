import os
import cv2
import base64
import json
import ssl
import numpy as np
import urllib.request
from flask import Flask, render_template, request, jsonify
from PIL import Image
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to allow self-signed HTTPS
def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)

# Function to classify an image using Roboflow API
def classify(image):
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="kTvcs84zAmxn8aoKdpZ1"
    )
    result = CLIENT.infer(image, model_id="wood-detection-ct5yx/1")
    return result

# Function to check if wood is detected
def check(result):
    if 'predictions' in result and len(result['predictions']) > 0:
        return True
    else:
        return False

# Function to crop the image based on prediction coordinates
def crop_image(image, prediction):
    try:
        x = int(prediction['x'])
        y = int(prediction['y'])
        width = int(prediction['width'])
        height = int(prediction['height'])
        cropped_image = image[y:y+height, x:x+width]
        return cropped_image
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Function to preprocess the cropped image using PIL and numpy
def preprocess_image(image):
    image_pil = Image.fromarray(image)
    image_resized = image_pil.resize((224, 224))
    image_array = np.array(image_resized)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 224, 224, 3)
    image_array = image_array.tolist()
    return image_array

# Function to send the preprocessed image to Azure for classification
def checkmodel(image):
    data = {'data': preprocess_image(image)}
    body = str.encode(json.dumps(data))
    url = 'http://9aa9d952-03a9-47cb-8608-ff37fb0f2bb5.centralindia.azurecontainer.io/score'
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(url, body, headers)
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    with open(image_path, 'rb') as f:
        image_data = f.read()

    image_base64 = base64.b64encode(image_data).decode('utf-8')
    image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)

    result = classify(image_base64)
    
    if check(result):
        cropped_image = crop_image(image_np, result['predictions'][0])
        if cropped_image is not None:
            prediction_result = checkmodel(cropped_image)
            if prediction_result:
                wood_prediction = prediction_result['prediction1']
                wood_class = int(np.argmax(wood_prediction))
                wood_class_name = "cipher" if wood_class == 0 else "pino patula"

                color_prediction = prediction_result['prediction2']
                color_class = int(np.argmax(color_prediction))

                return jsonify({
                    'wood_type': wood_class_name,
                    'color_code': color_class
                })
    return jsonify({'error': 'No wood detected '})

if __name__ == '__main__':
    app.run(debug=True)
