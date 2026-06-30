import os
import base64
from flask import Flask, request, jsonify, render_template
from predict import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def run_prediction():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Image comes as data:image/jpeg;base64,.....
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_frame.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
            
        score = predict(temp_path)
        
        return jsonify({'score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
