import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from predict import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    score = None
    filename = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
            
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Run prediction
                score = predict(filepath)
                result = "SCREEN / FAKE" if score >= 0.5 else "REAL PHOTO"
            except Exception as e:
                return render_template('index.html', error=f'Error processing image: {str(e)}')
                
    return render_template('index.html', result=result, score=score, filename=filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
