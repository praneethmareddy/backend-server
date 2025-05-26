from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from src.retrieve_and_respond import retrieve_and_respond

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'xlsx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "CIQ Standardization and Query System"

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Get query from form or default
                query = request.form.get('query', 'Standardize this CIQ file')
                
                # Process the file
                result = retrieve_and_respond(query, file_path)
                return jsonify(result)
            else:
                return jsonify({
                    "type": "error",
                    "message": "Invalid file type. Only Excel (.xlsx) files are allowed."
                }), 400
        
        # Handle text-only query
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "type": "error",
                "message": "No query provided"
            }), 400
            
        query = data['query']
        result = retrieve_and_respond(query)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "type": "error",
            "message": f"Server error: {str(e)}"
        }), 500

@app.route('/download', methods=['GET'])
def download_file():
    file_path = request.args.get('path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({
            "type": "error",
            "message": "File not found"
        }), 404
    
    try:
        return send_file(
            file_path,
            as_attachment=True,
            download_name=os.path.basename(file_path)
        )
    except Exception as e:
        return jsonify({
            "type": "error",
            "message": f"Error downloading file: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
