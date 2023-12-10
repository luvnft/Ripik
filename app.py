from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__, static_url_path='/static')
model = load_model('static/model_mobilenet.h5')

damage_types = {
    1: 'crack',
    2: 'scratch',
    3: 'tire flat',
    4: 'dent',
    5: 'glass shatter',
    6: 'lamp broken'
}

def process_image(file_storage):
    # Convert FileStorage to PIL Image
    img = Image.open(BytesIO(file_storage.read()))
    img = img.resize((224, 224))  # Resize to match the model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        img_array, img = process_image(file)
        predictions = model.predict(img_array)
        label = np.argmax(predictions)
        damage_type = damage_types.get(label + 1, 'Unknown')  # Adjust for 0-based indexing
        #result = f"Predicted Class: {label}, Damage Type: {damage_type}"
        result = f"Damage Type: {damage_type}"

        # Save the image temporarily in the static folder
        img.save('static/uploaded_image.png')

        # Render the template with the result and image path
        return render_template('index.html', result=result, image_path=url_for('static', filename='uploaded_image.png'))

    except Exception as e:
        return render_template('index.html', error=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
