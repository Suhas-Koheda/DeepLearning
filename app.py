import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import T5ForConditionalGeneration, RobertaTokenizer
import json
import logging

app = Flask(__name__)

# Configuration - Updated to use ML directory model path
MODEL_PATH = "/home/ssp/ML/model/codet5-fast"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for model and tokenizer
model = None
tokenizer = None
TASK_PREFIX = "Optimize Java: "
model_loaded = False
loading_error = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """Load the model and tokenizer"""
    global model, tokenizer, model_loaded, loading_error
    
    if model_loaded:
        return True
        
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        logger.info(f"Using device: {DEVICE}")
        
        # Check if model directory exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
        
        # Check for required files
        required_files = ['config.json', 'model.safetensors', 'tokenizer_config.json']
        for file in required_files:
            if not os.path.exists(os.path.join(MODEL_PATH, file)):
                raise FileNotFoundError(f"Required file not found: {file}")
        
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        
        model_loaded = True
        loading_error = None
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        loading_error = str(e)
        logger.error(f"Error loading model: {e}")
        model_loaded = False
        return False

def optimize_java_code(java_code):
    """Optimize Java code using the loaded model"""
    global model, tokenizer, model_loaded
    
    # Try to load model if not already loaded
    if not model_loaded:
        if not load_model():
            return f"Error: Model not loaded - {loading_error}"
    
    try:
        # Prepare input
        inputs = tokenizer(
            TASK_PREFIX + java_code.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(DEVICE)
        
        # Generate optimization
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode output
        optimized_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return optimized_code.strip()
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        return f"Error: {str(e)}"

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    """Handle code optimization requests"""
    try:
        data = request.get_json()
        java_code = data.get('code', '')
        
        if not java_code.strip():
            return jsonify({'error': 'No code provided'}), 400
        
        optimized_code = optimize_java_code(java_code)
        
        # Check if we got an error message
        if optimized_code.startswith("Error:"):
            return jsonify({'error': optimized_code}), 500
        
        return jsonify({
            'original': java_code,
            'optimized': optimized_code
        })
    except Exception as e:
        logger.error(f"Error in optimize endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model_loaded else 'loading' if not loading_error else 'error',
        'model_loaded': model_loaded,
        'loading_error': loading_error,
        'device': str(DEVICE),
        'model_path_exists': os.path.exists(MODEL_PATH)
    })

@app.route('/model-info')
def model_info():
    """Get model information"""
    info = {
        'model_path': MODEL_PATH,
        'model_path_exists': os.path.exists(MODEL_PATH),
        'device': str(DEVICE)
    }
    
    if os.path.exists(MODEL_PATH):
        try:
            files = os.listdir(MODEL_PATH)
            info['files'] = files
            info['model_size_mb'] = sum(os.path.getsize(os.path.join(MODEL_PATH, f)) for f in files if os.path.isfile(os.path.join(MODEL_PATH, f))) / (1024 * 1024)
        except Exception as e:
            info['error'] = str(e)
    
    return jsonify(info)

if __name__ == '__main__':
    # Try to load model on startup
    load_model()
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)