from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import psycopg2
from tensorflow import keras
import numpy as np
import os
from PIL import Image, ImageOps
from dotenv import load_dotenv
from psycopg2 import OperationalError
from psycopg2.extras import RealDictCursor
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_default_secret')  # Ensure this is set in your environment

# Load Keras model
model_path = "ChipAI/ChipAI/models/keras_model.h5"  # Update to your Keras model file
model = keras.models.load_model(model_path, compile=False)

# Load class labels from labels.txt
labels_path = "ChipAI/models/labels.txt"  # Path to your labels.txt
try:
    with open(labels_path, 'r') as f:
        class_labels = [line.strip() for line in f if line.strip()]
    logger.info("Loaded class labels: %s", class_labels)
except FileNotFoundError:
    logger.error("labels.txt not found at %s", labels_path)
    class_labels = ["Siling Atsal", "Siling Labuyo", "Siling Espada", "Scotch Bonnet", "Siling Talbusan"]  # Fallback
except Exception as e:
    logger.error("Error reading labels.txt: %s", str(e))
    class_labels = ["Siling Atsal", "Siling Labuyo", "Siling Espada", "Scotch Bonnet", "Siling Talbusan"]  # Fallback

# Ensure the uploads directory exists
upload_folder = 'Uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

load_dotenv()
def get_db_connection():
    try:
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT', '5432'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            dbname=os.getenv('DB_NAME')
        )
        logger.info("Database connection established.")
        return connection
    except OperationalError as e:
        logger.error("Database connection failed: %s", e)
        raise

# Mapping of chili names to image filenames
IMAGE_MAPPING = {
    "Siling Labuyo": "siling_labuyo.jpg",
    "Siling Atsal": "bell_pepper.jpg",
    "Siling Espada": "siling_haba.jpg", 
    "Scotch Bonnet": "scotch_bonnet.jpg",
    "Siling Talbusan": "siling_talbusan.jpg"
}

# Consolidated image preprocessing function matching Teachable Machine Keras
def preprocess_image(image_stream):
    try:
        img = Image.open(image_stream)  # Load image
        img = img.convert("RGB")  # Ensure RGB format
        size = (224, 224)
        img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)  # Crop to 224x224 from center
        img_array = np.asarray(img)  # Convert to numpy array
        normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1  # Normalize to [-1, 1]
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        logger.info("Image processed: shape=%s", data.shape)
        return data
    except Exception as e:
        logger.error("Error in preprocess_image: %s", str(e))
        return None

# Prediction function for the chili pepper classifier
def predict_chili_variety(image_stream):
    try:
        # Load and preprocess image
        data = preprocess_image(image_stream)
        if data is None:
            raise ValueError("Image loading failed")

        # Predict with Keras model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        predicted_prob = prediction[0][index]
        predicted_label = class_labels[index]
        confidence = float(predicted_prob)

        # Threshold for chili detection
        if predicted_prob < 0.50:
            logger.info("Prediction below threshold: %s (confidence: %.4f)", predicted_label, confidence)
            return {"label": "No Chili Detected", "confidence": confidence}

        logger.info("Prediction result: %s (confidence: %.4f)", predicted_label, confidence)
        return {"label": predicted_label, "confidence": confidence}
    except Exception as e:
        logger.error("Error in Keras prediction: %s", str(e))
        return {"label": "Error processing the image", "error": str(e)}

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        confirm_password = data.get('confirm_password')
        if not username or not password or not confirm_password:
            return jsonify({'success': False, 'message': 'All fields are required.'}), 400
        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match.'}), 400
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT sp_signup(%s, %s)", (username, password))
            result = cursor.fetchone()
            signup_status = result[0] if result else "Unknown error."
            if "success" in signup_status.lower():
                conn.commit()
                cursor.execute("SELECT * FROM sp_login(%s, %s)", (username, password))
                login_result = cursor.fetchone()
                if login_result and login_result[0]:
                    session['user_id'] = login_result[0]
                    logger.info("User signed up and logged in: %s", username)
                return jsonify({'success': True, 'message': signup_status.lower()}), 200
            else:
                return jsonify({'success': False, 'message': signup_status}), 400
        except Exception as e:
            logger.error("Signup error: %s", str(e))
            return jsonify({'success': False, 'message': str(e)}), 500
        finally:
            conn.close()
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password are required.'}), 400
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM sp_login(%s, %s)", (username, password))
            result = cursor.fetchone()
            if not result:
                return jsonify({'success': False, 'message': 'Login failed: No result returned.'}), 401
            p_user_id, _, p_status = result
            if p_status == 'User found' and p_user_id is not None:
                session['user_id'] = p_user_id
                logger.info("User logged in: %s", username)
                return jsonify({'success': True, 'message': 'login successful'}), 200
            else:
                return jsonify({'success': False, 'message': p_status}), 401
        except Exception as e:
            logger.error("Login error: %s", str(e))
            return jsonify({'success': False, 'message': f"Database error: {e}"}), 500
        finally:
            conn.close()
    # For GET requests, render login page if not logged in
    if 'user_id' not in session:
        logger.info("User not logged in, rendering login page")
        return render_template('index.html')
    logger.info("User already logged in, redirecting to dashboard")
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        logger.info("No user_id in session, redirecting to login")
        return redirect(url_for('login'))
    logger.info("Rendering dashboard for user_id: %s", session['user_id'])
    return render_template('login.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image_bytes = image.read()
        image_stream = BytesIO(image_bytes)
        img = Image.open(image_stream)
        img.verify()  # Verify image integrity
        logger.info("Image is valid.")
        image_stream.seek(0)

        # Predict chili variety
        result = predict_chili_variety(image_stream)
        if "error" in result:
            return jsonify({'error': f"Failed to process image: {result['error']}"}), 400

        return jsonify({
            'prediction': result['label'],
            'confidence': result['confidence']
        })
    except (IOError, SyntaxError) as e:
        logger.error("Invalid image file: %s", str(e))
        return jsonify({'error': 'Invalid image file. Please upload a valid image.'}), 400
    except Exception as e:
        logger.error("Error processing the image: %s", str(e))
        return jsonify({'error': 'Error processing the image. Please try again.'}), 500

@app.route('/')
def index():
    if 'user_id' in session:
        logger.info("User logged in, redirecting to dashboard")
        return redirect(url_for('dashboard'))
    logger.info("User not logged in, rendering index.html")
    return render_template('index.html')

@app.route('/ai')
def ai_model():
    if 'user_id' not in session:
        logger.info("User not logged in, redirecting to login")
        return redirect(url_for('login'))
    return render_template('AI.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/FAQs')
def faqs():
    if 'user_id' not in session:
        logger.info("User not logged in, redirecting to login")
        return redirect(url_for('login'))
    return render_template('faqs.html')

@app.route('/add_user', methods=['POST'])
def add_user():
    success = True
    if success:
        flash('User added successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    logger.info("User logged out")
    return redirect(url_for('index'))

@app.route('/get_chili_info', methods=['GET'])
def get_chili_info():
    chili_name = request.args.get('name')
    if not chili_name or chili_name in ["Error processing the image", "No Chili Detected"]:
        return jsonify({'error': 'Invalid chili name'}), 400
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT name, english_name, scientific_name, shu_range, description
            FROM chili_varieties WHERE name = %s
        """, (chili_name,))
        chili_info = cursor.fetchone()
        if not chili_info:
            return jsonify({'error': 'Chili not found'}), 404
        chili_info['image_url'] = url_for('static', filename=f'images/{IMAGE_MAPPING.get(chili_name, "default.jpg")}', _external=True)
        return jsonify(chili_info)
    finally:
        cursor.close()
        conn.close()

@app.route('/chili_trivia', methods=['GET'])
def chili_trivia():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT trivia_text FROM chili_trivia ORDER BY RANDOM() LIMIT 1")
        trivia = cursor.fetchone()
        if not trivia:
            return jsonify({'error': 'No trivia available'}), 404
        return jsonify(trivia)
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
