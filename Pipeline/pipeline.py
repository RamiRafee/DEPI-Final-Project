import mlflow
from data_loader.data_loader import load_data
from preprocessing.preprocessing import preprocess_data
from model_loader.model_loader import load_model
from inference.inference import run_inference

if __name__ == "__main__":
    # Set MLflow tracking URI 
    mlflow.set_tracking_uri("http://localhost:5000")  

    # Start a new MLflow run
    with mlflow.start_run():
        # Load dataset
        df = load_data()  

        # Preprocess data and generate YOLO labels
        preprocess_data(df)

        # Load YOLO model
        model = load_model()

        # Run inference
        run_inference(model)  # Run inference without expecting results to return

        # Log the number of samples processed
        mlflow.log_metric("inference_samples", len(df))  # Log the number of images processed
