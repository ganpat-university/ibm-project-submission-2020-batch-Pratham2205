from io import BytesIO
from flask import Flask, request, render_template, redirect, url_for, session, send_file, send_from_directory # type: ignore
from flask_dance.contrib.google import make_google_blueprint, google # type: ignore
from PIL import Image # type: ignore
from tensorflow.keras import layers, models # type: ignore
import hashlib
import os
import numpy as np # type: ignore
from fpdf import FPDF # type: ignore
from werkzeug.utils import secure_filename # type: ignore
from PyPDF2 import PdfReader, PdfWriter # type: ignore
import tempfile

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change this to a secure key

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'docx'}

ALLOWED_JPG_EXTENSIONS = {'jpg','png','jpeg'}

app.config['UPLOAD_FOLDER'] = 'uploads'

# Google OAuth configuration
google_blueprint = make_google_blueprint(
    client_id="965583567170-55aio9ke0p00foi97n9pn1n9qfb8pu85.apps.googleusercontent.com",
    client_secret="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",
    redirect_to="google_login",
    redirect_url="/login/google/callback",  # Callback route
    scope=["profile", "email"]
)
app.register_blueprint(google_blueprint, url_prefix="/login")

# Load the autoencoder model from the file or build if the file doesn't exist
autoencoder_model_file = 'autoencoder_model.h5'
autoencoder_history_file = 'autoencoder_history.npy'
if os.path.exists(autoencoder_model_file):
    autoencoder_model = models.load_model(autoencoder_model_file)
    history = np.load(autoencoder_history_file, allow_pickle='TRUE').item()
else:
    # Define input shape
    input_shape = (256, 256, 3)
    # Build the autoencoder model
    encoder_input = layers.Input(shape=input_shape)
    encoder = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    encoder = layers.MaxPooling2D((2, 2), padding='same')(encoder)
    bottleneck = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoder)
    decoder = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(bottleneck)
    decoder = layers.UpSampling2D((2, 2))(decoder)
    decoder_output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder)
    autoencoder_model = models.Model(encoder_input, decoder_output)
    autoencoder_model.compile(optimizer='adam', loss='mean_squared_error')
    history = None

# Function to load and preprocess the image
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((256, 256))  # Resize the image to a fixed size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return img_array

# Function to save the image temporarily
def save_image(img, filename):
    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    img_path = os.path.join(static_dir, filename)
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(img_path)
    return img_path

# Function to compress the image using the autoencoder
def compress_image(img, autoencoder_model):
    # Add batch dimension to the image
    img_batch = np.expand_dims(img, axis=0)
    # Compress the image using the autoencoder
    compressed_img = autoencoder_model.predict(img_batch)
    return compressed_img[0]

# Function to get image metadata including hash
def get_image_metadata(image_path):
    metadata = {}
    metadata['Hash'] = generate_file_hash(image_path)
    
    # Open the image using Pillow
    with Image.open(image_path) as img:
        # Get image dimensions
        width, height = img.size
        metadata['Resolution'] = f'{width}x{height}'
        
        # Get image mode (color space)
        metadata['Color Space'] = img.mode
        
        # Extract EXIF data
        exif = img.getexif()
        if exif:
            # Look for the creation date in EXIF data
            creation_date = exif.get(36867)  # 36867 corresponds to DateTimeOriginal tag
            if creation_date:
                metadata['Date Created'] = creation_date
                
    return metadata

# Function to generate file hash
def generate_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)  # Read file in chunks of 64KB
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()

# Route for the home page
@app.route('/')
def show_home():
    return render_template('home.html')

@app.route('/img-comp', methods=['GET', 'POST'])
def img_comp():
    global history

    original_img_path = None
    original_size_kb = None
    compressed_img_path = None
    compressed_size_kb = None
    original_metadata = None
    compressed_metadata = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('img-comp.html', error='No file uploaded')

        file = request.files['file']

        if file.filename == '':
            return render_template('img-comp.html', error='No file selected')

        if file and allowed_jpg_file(file.filename):  # Check if the file has a JPG extension
            img = load_image(file)

            # Train the autoencoder if history is not available
            if history is None:
                history = autoencoder_model.fit(np.expand_dims(img, axis=0), np.expand_dims(img, axis=0), epochs=10000, verbose=0)
                # Save the trained model and history
                if not os.path.exists(autoencoder_model_file):
                    autoencoder_model.save(autoencoder_model_file)
                if not os.path.exists(autoencoder_history_file):
                    np.save(autoencoder_history_file, history.history)

            # Compress the image using the autoencoder
            compressed_img = compress_image(img, autoencoder_model)
            
            original_img_path = save_image(img, 'uploaded_image.jpg')
            original_size = os.path.getsize(original_img_path)
            original_size_kb = original_size / 1024
            
            # Save the compressed image temporarily
            compressed_img_path = save_image(compressed_img, 'compressed_image.jpg')
            compressed_size = os.path.getsize(compressed_img_path)
            compressed_size_kb = compressed_size / 1024
            
            # Fetch metadata for both original and compressed images
            original_metadata = get_image_metadata(original_img_path)
            compressed_metadata = get_image_metadata(compressed_img_path)
            
    return render_template('img-comp.html', original_img_path=original_img_path, original_size=original_size_kb,
                           compressed_img_path=compressed_img_path, compressed_size=compressed_size_kb,
                           original_metadata=original_metadata, compressed_metadata=compressed_metadata)

@app.route('/download', methods=['POST'])
def download():
    compressed_img_path = request.form.get('compressed_img_path')
    if compressed_img_path:
        return send_file(compressed_img_path, as_attachment=True)
    else:
        return "No file to download"
def allowed_jpg_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_JPG_EXTENSIONS


#############################################################################################################
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to convert selected images into a PDF
def convert_to_pdf(image_files, output_pdf):
    pdf = FPDF()
    for image_file in image_files:
        pdf.add_page()
        pdf.image(image_file, 0, 0, 210, 297)  # Assuming A4 size (210x297 mm)
    pdf.output(output_pdf)

#############################################################################################################

@app.route('/convert-to-pdf-img', methods=['GET','POST'])
def convert_to_pdf_route():
    if request.method == 'GET':
        # Handle GET request (display form)
        return render_template('convert-to-pdf.html')
    elif request.method == 'POST':
        # Handle POST request (convert images to PDF)
        if 'files[]' not in request.files:
            return redirect(request.url)

        files = request.files.getlist('files[]')

        if len(files) == 0:
            return render_template('convert-to-pdf.html', error='No files selected')

        image_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                image_files.append(file_path)

        if len(image_files) == 0:
            return render_template('convert-to-pdf.html', error='No valid image files selected')

        output_pdf = 'output.pdf'
        output_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], output_pdf)
        convert_to_pdf(image_files, output_pdf_path)

        # Provide the relative path to the "uploads" folder for the download link
        return send_file(output_pdf_path, as_attachment=True)

#############################################################################################################

@app.route('/protect_pdf', methods=['GET', 'POST'])
def protect_pdf():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        uploaded_file = request.files['file']
        
        if uploaded_file.filename == '':
            return 'No selected file'
        
        if uploaded_file.filename.endswith('.pdf'):
            password = request.form.get('password')
            
            pdf_reader = PdfReader(uploaded_file)
            pdf_writer = PdfWriter()
            
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)
            
            # Encrypt the PDF with the provided password and save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                pdf_writer.encrypt(password)
                pdf_writer.write(temp_file)
                temp_file_path = temp_file.name
            
            # Set the filename for download
            filename = 'protected_pdf.pdf'
            
            response = send_file(temp_file_path, as_attachment=True)
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
        else:
            return 'Unsupported file format'
    else:
        return render_template('protect-pdf.html')
    
@app.route('/about')
def about():
    return render_template('about-us.html')

# Google OAuth callback route
@app.route("/login/google/callback")
def google_callback():
    if not google.authorized:
        return redirect(url_for("google.login"))

    resp = google.authorized_response()
    if resp is None or resp.get("access_token") is None:
        app.logger.error("Access denied: unable to fetch access token")
        return "Access denied: unable to fetch access token"

    session["google_token"] = (resp["access_token"], "")

    # Fetch user info using the access token
    user_info = google.get("/oauth2/v2/userinfo")
    if not user_info.ok:
        app.logger.error("Failed to fetch user info from Google API")
        return "Failed to fetch user info from Google API"

    user_info_json = user_info.json()
    session["user_id"] = user_info_json.get("id")  
    if session["user_id"] is None:
        app.logger.error("No user ID found in user info")
        return "No user ID found in user info"

    
    return redirect(url_for("home"))

# Google login route
@app.route("/login")
def google_login():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v2/userinfo")
    assert resp.ok, resp.text
    user_info = resp.json()
    session["user_id"] = user_info["id"] 
    return redirect(url_for("home"))

# Logout route
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
