from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import cv2
from mtcnn import MTCNN  # NEW

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

model = load_model('models/best_cdcn_model_v3_elite.h5')
last_conv_layer_name = "conv2d_2"

# NEW: MTCNN face detector
face_detector = MTCNN()

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    result = None
    image_url = None
    filename = None
    is_fake = False

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('detect.html', result="No file uploaded.")

        file = request.files['image']
        if file.filename == '':
            return render_template('detect.html', result="No file selected.")

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # ✅ NEW: Detect face using MTCNN
            img = cv2.imread(filepath)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect_faces(img_rgb)

            if len(faces) == 0:
                os.remove(filepath)  # نحذف الصورة لأنها غير صالحة
                return render_template('detect.html', result="No face detected in the image.")

            # ✅ Continue with prediction
            img_array = preprocess_image(filepath)
            pred = model.predict(img_array)[0][0]  
            result = 'Prediction: Real Face' if pred >= 0.5 else 'Prediction: Fake Face'
            image_url = url_for('static', filename=f'uploads/{file.filename}')
            filename = file.filename
            is_fake = pred < 0.5

    return render_template('detect.html', 
                           result=result, 
                           image_url=image_url, 
                           filename=filename,
                           is_fake=is_fake)

@app.route('/gradcam', methods=['GET'])
def gradcam():
    image_filename = request.args.get('image_filename', None)
    if not image_filename:
        return render_template('gradcam_upload.html', result="No image specified.")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    if not os.path.exists(filepath):
        return render_template('gradcam_upload.html', result="Image not found.")

    img_array = preprocess_image(filepath)
    preds = model.predict(img_array)
    pred_class = "Real Face" if preds[0][0] >= 0.5 else "Fake Face"
    confidence = preds[0][0] if preds[0][0] >= 0.5 else 1 - preds[0][0]

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    img_orig = cv2.imread(filepath)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap_colored, 0.4, 0)

    result_filename = 'gradcam_' + image_filename
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    image_url = url_for('static', filename=f'uploads/{image_filename}')
    gradcam_url = url_for('static', filename=f'results/{result_filename}')
    result = f"{pred_class} with confidence {confidence:.2%}"

    return render_template('gradcam_result.html',
                           result=result,
                           image_url=image_url,
                           gradcam_url=gradcam_url)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
