import onnx
import onnxruntime as ort

def load_model(model_path: str = './model_loader/best.onnx'):
    """Load a YOLOv5 model from an ONNX file."""
    
    # Load the ONNX model
    model = onnx.load(model_path)
    ort_session = ort.InferenceSession(model_path)

    print(f"Model loaded from {model_path}")
    
    return ort_session
