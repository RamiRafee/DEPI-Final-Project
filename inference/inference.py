import numpy as np
import cv2
import os
from glob import glob
import mlflow

# Constants for YOLO input size
INPUT_WIDTH = 640
INPUT_HEIGHT = 640



def preprocess_image(image_path):
    """Load and resize image for YOLOv5 model."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_detections(img, net):
    """Get predictions from YOLO model."""
    image = img.copy()
    row, col, _ = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # Prepare the image for the model
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_suppression(input_image, detections):
    """Filter detections based on confidence and apply non-maximum suppression."""
    boxes = []
    confidences = []
    
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for row in detections:
        confidence = row[4]  # Confidence score
        if confidence > 0.4:  # Confidence threshold
            class_score = row[5]  # Class score
            if class_score > 0.25:  # Class score threshold
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                
                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    
    # Apply Non-Maximum Suppression
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()
    
    return boxes_np, confidences_np, index

def load_actual_results(image_path):
    """Load actual bounding boxes from a corresponding text file."""
    actual_boxes = []
    # Assuming the actual results are stored in the same directory as images
    actual_file_path = image_path.replace('.jpg', '.txt')  # Change this to match your format
    if os.path.exists(actual_file_path):
        with open(actual_file_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 4:  # Assuming format: class_id confidence x1 y1 x2 y2
                    class_id, x1, y1, x2, y2 = map(float, parts)
                    actual_boxes.append((class_id, int(x1), int(y1), int(x2), int(y2)))
    return actual_boxes

def draw_boxes(image, boxes, color, label=None):
    """Draw bounding boxes on the image."""
    for box in boxes:
        if label:
            class_id, confidence, x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            print(box)
            x1, y1, width, height = box
            cv2.rectangle(image, (x1, y1), (x1 + width, y1 + height), color, 2)

def run_inference(model, data_folder: str = './yolov5/data_images/train') -> None:
    """Run inference on the provided data folder using the loaded YOLO model."""
    
    # Ensure the inference results directory exists
    os.makedirs('./inference_results/', exist_ok=True)

    # Get all image paths
    image_paths = glob(os.path.join(data_folder, '*.jpg'))
    
    # Run inference on the first 10 images
    for image_path in image_paths[:10]:
        # Preprocess the image
        img = preprocess_image(image_path)

        # Get detections
        input_image, detections = get_detections(img, model)

        # Apply non-maximum suppression
        boxes, confidences, indices = non_maximum_suppression(input_image, detections)

        # Load actual results
        actual_boxes = load_actual_results(image_path)

        # Draw predicted boxes in green and actual boxes in red
        draw_boxes(input_image, [boxes[i] for i in indices], color=(0, 255, 0))  # Predicted
        draw_boxes(input_image, actual_boxes, color=(0, 0, 255))  # Actual

        # Save annotated image
        annotated_image_path = os.path.join('./inference_results/', os.path.basename(image_path))
        cv2.imwrite(annotated_image_path, input_image)

        # Prepare results for saving
        output_file = os.path.join('./inference_results/', os.path.basename(image_path).replace('.jpg', '.txt'))
        with open(output_file, 'w') as f:
            for i in indices:
                box = boxes[i]
                x1, y1, width, height = box
                x2, y2 = x1 + width, y1 + height
                confidence = confidences[i]
                class_id = 0  # Assuming single class; update if multiple classes are used
                f.write(f"{class_id} {confidence:.2f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

        # Log the image and results
        mlflow.log_artifact(annotated_image_path)  # Log the annotated image
        mlflow.log_artifact(output_file)  # Log the result file
    
    # Log the number of images processed
    mlflow.log_metric("inference_samples", len(image_paths[:10]))
