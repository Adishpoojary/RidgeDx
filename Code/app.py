"""
RidgeDx — Flask Backend
========================
AI-powered blood group prediction from fingerprint images.

Features:
    - ResNet-18 model (grayscale, 8-class)
    - Grad-CAM explainability heatmaps
    - REST API v1 with Swagger docs
    - Session-based auth with SQLite
    - Scan history tracking
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import time
import base64
import io
import sqlite3
import logging
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

from config import get_config

# Conditional imports for Grad-CAM and Swagger
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logging.warning("pytorch-grad-cam not installed. Grad-CAM will be disabled.")

try:
    from flask_restx import Api, Resource, Namespace, fields
    SWAGGER_AVAILABLE = True
except ImportError:
    SWAGGER_AVAILABLE = False
    logging.warning("flask-restx not installed. Swagger docs will be disabled.")


# ============================================================
# Flask App Setup
# ============================================================
app = Flask(__name__, template_folder='templates')
config = get_config()
app.config.from_object(config)

CORS(app, origins=config.CORS_ORIGINS)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

logging.basicConfig(
    level=logging.DEBUG if app.config.get('DEBUG') else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('ridgedx')


# ============================================================
# Database Setup
# ============================================================
def get_db_connection():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fullname TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT,
                predicted_group TEXT NOT NULL,
                confidence REAL NOT NULL,
                inference_ms REAL NOT NULL,
                all_probabilities TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()


init_db()


# ============================================================
# Model: ResNet-18 Modified for Grayscale + 8 Classes
# ============================================================
def build_resnet18(num_classes=8, pretrained=False):
    """Build ResNet-18 modified for 1-channel grayscale input."""
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)

    # Modify first conv layer: 3-channel -> 1-channel
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # If pretrained, average the 3-channel weights to 1-channel
    if pretrained:
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

    # Modify final FC layer for 8 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# ============================================================
# Load Model
# ============================================================
num_classes = config.NUM_CLASSES
blood_groups = config.BLOOD_GROUPS
model = build_resnet18(num_classes=num_classes)

model_path = config.MODEL_PATH
if os.path.exists(model_path):
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    )
    logger.info(f"Loaded ResNet-18 model from {model_path}")
else:
    # Try legacy model
    legacy_path = os.path.join(config.BASE_DIR, 'new_model_testing.pkl')
    if os.path.exists(legacy_path):
        logger.warning(
            f"ResNet-18 weights not found at {model_path}. "
            f"Legacy SimpleCNN weights at {legacy_path} are incompatible. "
            f"Please train a new model with train_model_v2.py"
        )
    else:
        logger.warning("No model weights found. Predictions will use random weights.")

model.eval()


# ============================================================
# Image Preprocessing
# ============================================================
def transform_image(img):
    """Preprocess image for model inference.

    Pipeline: Grayscale -> Resize(128x128) -> Normalize to [0,1] -> Tensor
    """
    img = img.convert('L')
    img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
    return tensor


# ============================================================
# Grad-CAM
# ============================================================
def generate_gradcam(model, input_tensor, original_img):
    """Generate Grad-CAM heatmap for the prediction.

    Returns base64-encoded PNG image of the heatmap overlay.
    """
    if not GRADCAM_AVAILABLE:
        return None

    try:
        target_layer = model.layer4[-1]
        cam = GradCAM(model=model, target_layers=[target_layer])

        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]

        # Create RGB version of original for overlay
        original_resized = original_img.convert('L').resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE)
        )
        rgb_img = np.array(original_resized, dtype=np.float32) / 255.0
        rgb_img = np.stack([rgb_img] * 3, axis=-1)

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Convert to base64
        pil_img = Image.fromarray(visualization)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        cam.__del__()
        return b64

    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}")
        return None


# ============================================================
# Swagger API (flask-restx)
# ============================================================
if SWAGGER_AVAILABLE:
    api = Api(
        app,
        version='1.0',
        title='RidgeDx API',
        description='AI-powered blood group prediction from fingerprint images',
        doc='/api/docs',
        prefix='/api/v1'
    )

    predict_ns = Namespace('predict', description='Prediction operations')
    api.add_namespace(predict_ns, path='/api/v1')

    prediction_model = api.model('PredictionResult', {
        'prediction': fields.String(description='Predicted blood group', example='A+'),
        'confidence': fields.Float(description='Confidence score (0-100)', example=92.5),
        'all_probabilities': fields.Raw(description='Probabilities for all 8 blood groups'),
        'grad_cam_base64': fields.String(description='Base64-encoded Grad-CAM heatmap PNG'),
        'inference_time_ms': fields.Float(description='Model inference time in milliseconds', example=42.3),
    })

    @predict_ns.route('/predict')
    class PredictResource(Resource):
        @predict_ns.doc('predict_blood_group')
        @predict_ns.marshal_with(prediction_model)
        def post(self):
            """Predict blood group from a fingerprint image."""
            return _handle_api_predict()

else:
    # Fallback: plain Flask route if flask-restx not installed
    @app.route('/api/v1/predict', methods=['POST'])
    def api_predict():
        return jsonify(_handle_api_predict())


def _handle_api_predict():
    """Core prediction logic shared by Swagger and fallback routes."""
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': 'No file selected'}, 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load and preprocess
        original_img = Image.open(file_path)
        img_tensor = transform_image(original_img).unsqueeze(0)  # (1, 1, H, W)

        # Inference
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        inference_ms = (time.perf_counter() - start_time) * 1000

        # Results
        predicted_idx = torch.argmax(probabilities).item()
        predicted_group = blood_groups[predicted_idx]
        confidence = probabilities[predicted_idx].item() * 100

        all_probs = {
            blood_groups[i]: round(probabilities[i].item() * 100, 2)
            for i in range(num_classes)
        }

        # Grad-CAM
        gradcam_b64 = generate_gradcam(model, img_tensor, original_img)

        # Save to scan history if user is logged in
        if 'user_id' in session:
            try:
                import json
                conn = get_db_connection()
                conn.execute(
                    '''INSERT INTO scans (user_id, filename, predicted_group,
                       confidence, inference_ms, all_probabilities)
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (session['user_id'], filename, predicted_group,
                     confidence, inference_ms, json.dumps(all_probs))
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to save scan: {e}")

        result = {
            'prediction': predicted_group,
            'confidence': round(confidence, 2),
            'all_probabilities': all_probs,
            'grad_cam_base64': gradcam_b64,
            'inference_time_ms': round(inference_ms, 2)
        }

        logger.info(
            f"Prediction: {predicted_group} ({confidence:.1f}%) "
            f"in {inference_ms:.1f}ms"
        )

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {'error': str(e)}, 500


# ============================================================
# Legacy Predict Route (renders template)
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = Image.open(file_path).convert('L')
        img_tensor = transform_image(img).unsqueeze(0)

        with torch.no_grad():
            prediction = model(img_tensor)

        predicted_idx = torch.argmax(prediction).item()
        predicted_group = blood_groups[predicted_idx]

        return render_template('result.html', blood_group=predicted_group)


# ============================================================
# Page Routes
# ============================================================
@app.route('/')
def index():
    return render_template('index_new.html')


@app.route('/home')
def home():
    return render_template('index_new.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            return redirect(url_for('predict_blood_group'))
        else:
            flash('Invalid username or password!', 'error')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']

        if password != confirmpassword:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (fullname, email, username, password) VALUES (?, ?, ?, ?)',
                (fullname, email, username, hashed_password)
            )
            conn.commit()
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
        finally:
            conn.close()
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('Please log in to access your profile.', 'error')
        return redirect(url_for('login'))

    conn = get_db_connection()
    user_id = session['user_id']

    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()

    if user is None:
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']

        if not fullname or not email or not username:
            flash('All fields are required.', 'error')
            return redirect(url_for('profile'))

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE users SET fullname = ?, email = ?, username = ? WHERE id = ?',
                (fullname, email, username, user_id)
            )
            conn.commit()
            conn.close()
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))
        except Exception:
            flash('An error occurred while updating your profile.', 'error')
            return redirect(url_for('profile'))

    return render_template('profile.html', user=user)


@app.route('/predict_blood_group')
def predict_blood_group():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    return render_template('login2.html')


# ============================================================
# API: Scan History
# ============================================================
@app.route('/api/v1/scans', methods=['GET'])
def get_scans():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    conn = get_db_connection()
    scans = conn.execute(
        'SELECT * FROM scans WHERE user_id = ? ORDER BY created_at DESC LIMIT 50',
        (session['user_id'],)
    ).fetchall()
    conn.close()

    return jsonify([dict(s) for s in scans])


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':
    app.run(debug=app.config.get('DEBUG', True))
