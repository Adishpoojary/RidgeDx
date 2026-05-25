"""
RidgeDx — Lightweight Deployment Backend (ONNX Runtime)
========================================================
Uses ONNX Runtime instead of PyTorch for 6x lower memory usage.
Designed for Render free tier (512MB RAM).
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import time
import sqlite3
import logging
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

from config import get_config

# Conditional Swagger import
try:
    from flask_restx import Api, Resource, Namespace, fields
    SWAGGER_AVAILABLE = True
except ImportError:
    SWAGGER_AVAILABLE = False

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
# ONNX Runtime Model
# ============================================================
num_classes = config.NUM_CLASSES
blood_groups = config.BLOOD_GROUPS

onnx_path = os.path.join(config.BASE_DIR, 'resnet18_bloodgroup.onnx')
if os.path.exists(onnx_path):
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    logger.info(f"Loaded ONNX model from {onnx_path}")
else:
    ort_session = None
    logger.warning(f"ONNX model not found at {onnx_path}")


# ============================================================
# Image Preprocessing
# ============================================================
def transform_image(img):
    """Preprocess image: Grayscale -> Resize(128x128) -> Normalize -> numpy array"""
    img = img.convert('L')
    img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    # ONNX expects (batch, channels, height, width)
    return img_array.reshape(1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)


# ============================================================
# Softmax
# ============================================================
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ============================================================
# Swagger API
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
        'inference_time_ms': fields.Float(description='Inference time in ms', example=42.3),
    })

    @predict_ns.route('/predict')
    class PredictResource(Resource):
        @predict_ns.doc('predict_blood_group')
        @predict_ns.marshal_with(prediction_model)
        def post(self):
            """Predict blood group from a fingerprint image."""
            return _handle_api_predict()
else:
    @app.route('/api/v1/predict', methods=['POST'])
    def api_predict():
        return jsonify(_handle_api_predict())


def _handle_api_predict():
    """Core prediction logic using ONNX Runtime."""
    if ort_session is None:
        return {'error': 'Model not loaded'}, 503

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
        input_array = transform_image(original_img)

        # ONNX Runtime inference
        start_time = time.perf_counter()
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_array})
        inference_ms = (time.perf_counter() - start_time) * 1000

        # Results
        logits = outputs[0][0]
        probabilities = softmax(logits)
        predicted_idx = int(np.argmax(probabilities))
        predicted_group = blood_groups[predicted_idx]
        confidence = float(probabilities[predicted_idx]) * 100

        all_probs = {
            blood_groups[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(num_classes)
        }

        # Save scan history
        if 'user_id' in session:
            try:
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
            'grad_cam_base64': None,
            'inference_time_ms': round(inference_ms, 2)
        }

        logger.info(f"Prediction: {predicted_group} ({confidence:.1f}%) in {inference_ms:.1f}ms")
        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {'error': str(e)}, 500


# ============================================================
# Legacy Predict Route
# ============================================================
@app.route('/predict', methods=['POST'])
def predict():
    if ort_session is None:
        return jsonify({'error': 'Model not loaded'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = Image.open(file_path)
        input_array = transform_image(img)

        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_array})
        predicted_idx = int(np.argmax(outputs[0][0]))
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
