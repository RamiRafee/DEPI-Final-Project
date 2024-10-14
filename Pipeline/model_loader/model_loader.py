
import cv2


def load_model(model_path: str = './model_loader/best.onnx'):
    """Load the YOLO model from the specified ONNX file."""
    # Load the model
    net = cv2.dnn.readNetFromONNX(model_path)
    
    # Set preferable backend and target
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    return net
