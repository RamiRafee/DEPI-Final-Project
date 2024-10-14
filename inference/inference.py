import numpy as np
import cv2
import os
from glob import glob
import mlflow

def preprocess_image(image_path):
    """Preprocess the image for YOLOv5 model."""
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize the image
    img_resized = cv2.resize(img, (640, 640))  # Resize to YOLOv5 input size
    img_normalized = img_resized / 255.0  # Normalize to [0, 1]
    
    # Transpose the image to have channels first
    img_transposed = np.transpose(img_normalized, (2, 0, 1)).astype(np.float32)
    
    return img_transposed

def run_inference(model, data_folder: str = './yolov5/data_images/train') -> None:
    """Run inference on the provided data folder using the loaded ONNX model."""
    
    # Ensure the inference results directory exists
    os.makedirs('./inference_results/', exist_ok=True)

    # Get all image paths
    image_paths = glob(os.path.join(data_folder, '*.jpg'))
    
    # Run inference on the first 10 images
    for image_path in image_paths[:10]:
        # Preprocess the image
        img = preprocess_image(image_path)
        
        # Run inference
        results = model.run(None, {model.get_inputs()[0].name: img[None, :]})  # Add batch dimension
        
        # Debugging output
        print("Inference results:", results)
        
        # Check the results length and handle accordingly
        if len(results) >= 3:
            boxes, scores, class_ids = results[0], results[1], results[2]
        else:
            print("Unexpected results format. Results length:", len(results))
            continue  # Skip to the next image

        # Prepare results for saving
        output_file = os.path.join('./inference_results/', os.path.basename(image_path).replace('.jpg', '.txt'))
        with open(output_file, 'w') as f:
            for box, score, class_id in zip(boxes, scores, class_ids):
                if score > 0.5:  # Threshold for detection
                    # Assuming box format is [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box
                    f.write(f"{class_id} {score:.2f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

        # Log the image and results
        mlflow.log_artifact(image_path)  # Log the input image
        mlflow.log_artifact(output_file)  # Log the result file
    
    # Log the number of images processed
    mlflow.log_metric("inference_samples", len(image_paths[:10]))
