"""
app.py
Flask web application for handwriting recognition
"""

from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import io
import base64
from werkzeug.utils import secure_filename
import os

# Import the recognizer from the main module
from handwriting_recognition import HandwritingRecognizer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize recognizer with trained model
recognizer = None

def init_recognizer():
    """Initialize the recognizer with trained model"""
    global recognizer
    recognizer = HandwritingRecognizer('emnist_model.pkl')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'png', 'bmp', 'jpg', 'jpeg'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Use PNG or BMP'}), 400
        
        # Read image
        image = Image.open(file.stream).convert('L')  # Convert to grayscale
        
        # Resize to 28x28 if needed
        if image.size != (28, 28):
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Invert if needed (EMNIST expects white text on black background)
        # Check if background is lighter than foreground
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Get prediction
        result = recognizer.predict(img_array)
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'character': result['character'],
            'confidence': f"{result['confidence']:.2%}",
            'top5': result['top5'],
            'image': img_base64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_drawing', methods=['POST'])
def predict_drawing():
    """Handle canvas drawing prediction"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')
        
        # Convert to numpy array
        img_array = np.array(image)

        # Find bounding box of the drawing to crop and center it
        rows = np.any(img_array > 30, axis=1)
        cols = np.any(img_array > 30, axis=0)

        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Add padding (20% of size)
            height = rmax - rmin
            width = cmax - cmin
            pad_h = int(height * 0.2)
            pad_w = int(width * 0.2)

            rmin = max(0, rmin - pad_h)
            rmax = min(img_array.shape[0], rmax + pad_h)
            cmin = max(0, cmin - pad_w)
            cmax = min(img_array.shape[1], cmax + pad_w)

            # Crop to bounding box
            cropped = img_array[rmin:rmax, cmin:cmax]

            # Create 28x28 image with the character centered
            # Make it square first
            h, w = cropped.shape
            size = max(h, w)
            square = np.zeros((size, size), dtype=np.uint8)

            # Center the cropped image in the square
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

            # Resize to 28x28
            image = Image.fromarray(square)
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(image)
        else:
            # If no drawing detected, just resize
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(image)

        # The canvas already has white on black, so no inversion needed
        # But ensure proper contrast
        img_array = img_array.astype(np.float32)

        # Normalize and enhance contrast
        if img_array.max() > 0:
            img_array = (img_array / img_array.max()) * 255.0
        img_array = img_array.astype(np.uint8)
        
        # Get prediction
        result = recognizer.predict(img_array)
        
        return jsonify({
            'success': True,
            'character': result['character'],
            'confidence': f"{result['confidence']:.2%}",
            'top5': [(char, f"{conf:.2%}") for char, conf in result['top5']]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create upload folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize recognizer
    print("Loading model...")
    init_recognizer()
    print("Model loaded successfully!")
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)