from flask import Flask,render_template, request, jsonify, send_from_directory
import datetime
import cv2
import numpy as np
import os
from Pipeline.model_loader.model_loader import load_model  # Update this to match your file structure
from Pipeline.inference.inference import preprocess_image, get_detections, non_maximum_suppression, draw_boxes  # Update this import accordingly

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH_NPR = os.path.join(BASE_PATH, 'static', 'upload', 'NPR')

ROI_PATH_NPR = os.path.join(BASE_PATH, 'static', 'roi', 'NPR')

PREDICT_PATH_NPR = os.path.join(BASE_PATH, 'static', 'predict', 'NPR')




model = load_model('Pipeline/model_loader/best.onnx')  # Load the YOLO model

@app.route('/', methods=['POST','GET'])
def index():
    """Handle file uploads."""
    if request.method == 'POST':
        if 'image_name_NPR' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['image_name_NPR']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded file
        image_path = os.path.join('static/inference_results/', file.filename)
        file.save(image_path)

        # Run inference
        annotated_image = run_inference(model, image_path)

        # Save the annotated image
        output_image_path = os.path.join('static/inference_results/', f'annotated_{file.filename}')
        cv2.imwrite(output_image_path, annotated_image)
        return render_template('index.html', NPR=True, upload_image=file.filename )
    return render_template('index.html', upload=False)

def run_inference(model, image_path):
    """Run inference on a single image and return the annotated image."""
    img = preprocess_image(image_path)
    input_image, detections = get_detections(img, model)
    boxes, confidences, indices = non_maximum_suppression(input_image, detections)
    
    # Draw bounding boxes on the image
    draw_boxes(input_image, [boxes[i] for i in indices], color=(0, 255, 0))  # Draw predicted boxes
    # Optionally, you could load actual results here and draw them as well
    
    return input_image

if __name__ == '__main__':
    os.makedirs('static/inference_results/', exist_ok=True)  # Ensure the directory exists
    app.run(debug=True)