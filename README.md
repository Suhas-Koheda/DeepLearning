# Java Code Optimizer

A web-based interface for optimizing Java code using a fine-tuned CodeT5-small model.

## Overview

This application provides a user-friendly web interface to optimize Java code using a fine-tuned Salesforce/codet5-small model. The model has been trained on Java optimization tasks and can transform verbose or inefficient Java code into more optimal versions.

## Features

- 🚀 **Real-time Optimization**: Get instant optimization suggestions for your Java code
- 🎯 **Specialized Model**: Fine-tuned specifically for Java code optimization tasks
- 💻 **GPU/CPU Support**: Automatically utilizes GPU if available, falls back to CPU
- 🎨 **Modern Interface**: Clean, responsive design with dark/light mode support
- 📚 **Example Cases**: Pre-loaded examples to demonstrate optimization capabilities
- 🔍 **Health Monitoring**: Real-time status of model loading and device utilization

## Model Information

- **Base Model**: Salesforce/codet5-small
- **Fine-tuned Model**: [nlpctx/codet5-java-optimizer](https://huggingface.co/nlpctx/codet5-java-optimizer)
- **Training Dataset**: [nlpctx/java_optimisation](https://huggingface.co/datasets/nlpctx/java_optimisation)
- **Training Data**: ~6K training / 680 validation Java optimization pairs
- **Framework**: HuggingFace Transformers with Seq2SeqTrainer
- **Optimization Focus**: Java code refactoring and performance improvements

## Installation & Setup

### Prerequisites

- Python 3.8+
- Git
- Internet connection (for initial model loading)

### Step-by-Step Instructions

1. **Clone/Navigate to the Project Directory**
   ```bash
   cd ~/model/java_optimizer
   ```

2. **Activate the Virtual Environment**
   ```bash
   source ~/Python/.venv/bin/activate
   ```

3. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Load the Model from Hugging Face**
    The application loads the fine-tuned model from Hugging Face:
    - **Model**: [nlpctx/codet5-java-optimizer](https://huggingface.co/nlpctx/codet5-java-optimizer)
    - **Dataset**: [nlpctx/java_optimisation](https://huggingface.co/datasets/nlpctx/java_optimisation)
    
    The model will be automatically downloaded on first run.

5. **Run the Application**
   ```bash
   python app.py
   ```

6. **Access the Web Interface**
   Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Enter Java Code**: Type or paste your Java code into the input textarea
2. **Click Optimize**: Press the "⚡ Optimize Code" button
3. **View Results**: The optimized code will appear in the output textarea
4. **Try Examples**: Click on any of the pre-loaded examples to test the optimizer

## Example Optimizations

The model has been trained to recognize and optimize common Java patterns:

- **Switch Expressions**: Converting verbose switch statements to switch expressions
- **Collection Operations**: Replacing manual iterator removal with `removeIf()`
- **String Handling**: Optimizing string concatenation with `StringBuilder`
- **And more...**

## API Endpoints

### GET `/`
- Returns the main HTML interface

### POST `/optimize`
- **Request Body**: `{ "code": "your Java code here" }`
- **Response**: 
  ```json
  {
    "original": "input Java code",
    "optimized": "optimized Java code"
  }
  ```
- **Error Response**: `{ "error": "error message" }`

### GET `/health`
- Returns application health status including model loading state and device info

### GET `/model-info`
- Returns information about the model files and size

## Troubleshooting

### Common Issues

1. **Model Not Loading**
    - Ensure you have an internet connection for model downloading
    - Verify Hugging Face credentials if using gated models
    - Check that you have sufficient disk space and memory

2. **CUDA/GPU Issues**
   - The application will automatically fall back to CPU if GPU is unavailable
   - To force CPU usage, modify the `DEVICE` variable in `app.py`

3. **Port Already in Use**
   - Change the port in `app.py` (line with `app.run()`)
   - Or stop the existing process using port 5000

4. **Dependencies Missing**
   - Run `pip install torch transformers flask` manually
   - Check Python version compatibility

### Performance Notes

- First optimization may be slower due to model loading
- Subsequent requests are faster as model stays in memory
- GPU acceleration significantly improves inference speed
- Model size is approximately 240MB

## Development

### File Structure
```
java_optimizer
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── test_model.py         # Model testing script
└── templates/
    └── index.html        # Frontend interface
```

### Making Changes
1. Modify `app.py` for backend logic changes
2. Update `templates/index.html` for UI changes
3. Adjust model parameters in `app.py` if needed
4. Restart the application after changes

## License

This project is provided for educational and demonstration purposes.

## Acknowledgements

- Model based on Salesforce/codet5-small
- Fine-tuned model: [nlpctx/codet5-java-optimizer](https://huggingface.co/nlpctx/codet5-java-optimizer)
- Training data: [nlpctx/java_optimisation](https://huggingface.co/datasets/nlpctx/java_optimisation)
- Built with Flask and HuggingFace Transformers
