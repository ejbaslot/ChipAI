from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import mysql.connector
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_default_secret')  # Change for production

# Load the updated model
model = tf.keras.models.load_model('ChipAI/models/chili_pepper_classifier.h5')  # Ensure the correct model path

# Ensure the uploads directory exists
upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Database connection function
def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='ChipAI'
    )
    return connection

# Mapping of chili names to image filenames (No need to store images in DB)
IMAGE_MAPPING = {
    "Siling Labuyo": "siling_labuyo.jpg",
    "Siling Atsal": "bell_pepper.jpg",
    "Siling Espada": "siling_haba.jpg",
    "Siling Demonyo": "siling_demonyo.jpg"
}

# Image preprocessing function (handling both in-memory and file path)
def preprocess_image(image_stream):
    try:
        # Open the image from the in-memory stream
        img = Image.open(image_stream)
        img = img.resize((150, 150))  # Resize the image to match the model input size
        img = np.array(img)  # Convert image to numpy array
        img = img / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        return None
    
# Prediction function for the chili pepper classifier
def predict_chili_variety(image_stream):
    try:
        # Define image parameters
        IMG_HEIGHT, IMG_WIDTH = 150, 150

        # Load and preprocess the image from stream
        img = Image.open(image_stream).resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to fit model input shape

        # Predict using the model
        prediction = model.predict(img_array)

        # Get the predicted label with the highest probability
        class_labels = ["Siling Atsal", "Siling Labuyo", "Siling Espada", "Siling Demonyo"]
        predicted_prob = np.max(prediction)  # Highest predicted probability
        predicted_label = class_labels[np.argmax(prediction)]  # Corresponding class label

        # Check confidence threshold (e.g., 50% confidence)
        if predicted_prob < 0.50:
            return "No Chili Detected"

        return predicted_label
    except Exception as e:
        print(f"Error in prediction function: {str(e)}")
        return "Error processing the image."

# Route for signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('signup'))

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # Call the stored procedure for signup
            cursor.callproc('sp_signup', (username, password, '@p_status'))
            conn.commit()

            # Check for status message from the stored procedure
            cursor.execute('SELECT @p_status')
            signup_status = cursor.fetchone()[0]

            if signup_status == 'Signup successful':
                flash(signup_status, 'success')
                return redirect(url_for('login'))
            else:
                flash(signup_status, 'error')
                return redirect(url_for('signup'))
        except Exception as e:
            flash(str(e), 'error')
            return redirect(url_for('signup'))
        finally:
            conn.close()

    return render_template('login.html')

# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # Call the stored procedure for login
            cursor.callproc('sp_login', (username, password, 0, '', '@p_status'))
            conn.commit()

            # Retrieve the results from the stored procedure
            cursor.execute('SELECT @p_status, @p_user_id')
            result = cursor.fetchone()
            login_status, user_id = result

            if login_status == 'User found' and user_id > 0:
                session['user_id'] = user_id
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                flash(login_status, 'error')  # Use the login status as the error message
                return redirect(url_for('login'))
        except Exception as e:
            flash(str(e), 'error')
            return redirect(url_for('login'))
        finally:
            conn.close()

    return render_template('login.html')

# Route for uploading and processing image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    image = request.files['image']
    
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Check if the file is a valid image
        img = Image.open(image.stream)  # Open the image from the stream
        img.verify()  # Verify that it's an image file

        # Step 2: Preprocess the image
        preprocessed_image = preprocess_image(image.stream)

        # Step 3: Make prediction after preprocessing
        prediction = model.predict(preprocessed_image)

        # Check the model output and return appropriate label
        predicted_label = predict_chili_variety(image.stream)

        # Return the prediction result as JSON
        return jsonify({'prediction': predicted_label})
    except (IOError, SyntaxError) as e:
        print(f"Invalid image file: {str(e)}")
        return jsonify({'error': 'Invalid image file. Please upload a valid image.'}), 400
    except Exception as e:
        print(f"Error processing the image: {str(e)}")
        return jsonify({'error': 'Error processing the image. Please try again.'}), 500

# Route for homepage (index)
@app.route('/')
def index():
    user_id = session.get('user_id')
    return render_template('index.html', user_id=user_id)

@app.route('/ai')
def ai_model():
    return render_template('AI.html')  # Ensure this matches the file name exactly

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/FAQs')
def faqs():
    return render_template('faqs.html')


@app.route('/add_user', methods=['POST'])
def add_user():
    # Your code to add the user
    success = True  # Assuming the user is added successfully
    if success:
        flash('User added successfully!', 'success')
    return redirect('/index.html')

# Route for logout
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/get_chili_info', methods=['GET'])
def get_chili_info():
    chili_name = request.args.get('name')  # Get prediction name from request
    if not chili_name:
        return jsonify({'error': 'Chili name is required'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = """
    SELECT name, english_name, scientific_name, shu_range, description
    FROM chili_varieties WHERE name = %s
    """
    cursor.execute(query, (chili_name,))
    chili_info = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    if not chili_info:
        return jsonify({'error': 'Chili not found'}), 404

    # Assign image URL dynamically
    chili_info['image_url'] = url_for('static', filename=f'images/{IMAGE_MAPPING.get(chili_name, "default.jpg")}', _external=True)

    return jsonify(chili_info)

@app.route('/chili_trivia', methods=['GET'])
def chili_trivia():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = "SELECT trivia_text FROM chili_trivia ORDER BY RAND() LIMIT 1"
    cursor.execute(query)
    trivia = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    if not trivia:
        return jsonify({'error': 'No trivia available'}), 404

    return jsonify(trivia)

if __name__ == '__main__':
    app.run()
