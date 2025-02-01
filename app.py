from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from SNN import predict_genre  # Import the function from SNN.py

app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# print(f"Uploaded file saved at: {"C:\Users\siddh\OneDrive\Desktop\Genre\uploads"})

@app.route('/')
def home():
    # Render the HTML page with the upload form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # File upload validation
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict genre
        genre = predict_genre(file_path)
        return render_template('index.html', genre=genre)
    except Exception as e:
        # Handle errors
        return render_template('index.html', error=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
