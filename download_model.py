import os
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

def download_hockey_model():
    """
    Download a hockey-specific YOLO model from Hugging Face repository
    """
    print("Downloading hockey-specific detection model from Hugging Face...")

    # Create models directory if it doesn't exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Try multiple hockey-related repositories
    model_configs = [
        {"repo": "SimulaMet-HOST/HockeyAI", "file": "HockeyAI_model_weight.pt"},
        {"repo": "yolo-rodri/hockey-yolo", "file": "weights/best.pt"},
        {"repo": "gligame/hockey-detector", "file": "weights/best.pt"},
    ]

    for config in model_configs:
        try:
            print(f"Trying to download from {config['repo']}/{config['file']}...")
            model_path = hf_hub_download(
                repo_id=config["repo"],
                filename=config["file"],
                local_dir=model_dir
            )
            print(f"Model downloaded successfully from {config['repo']} to: {model_path}")
            break
        except Exception as e:
            print(f"Failed to download from {config['repo']}: {e}")
            continue
    else:
        # If all hockey models fail, download a basic model
        print("No hockey-specific models found. Downloading basic YOLOv8 model...")
        model = YOLO('yolov8n.pt')  # Download the basic model
        basic_model_path = os.path.join(model_dir, "basic_yolo.pt")
        model.save(basic_model_path)
        model_path = basic_model_path
        print(f"Basic YOLO model saved to: {basic_model_path}")

    # Verify the model can be loaded
    print("Verifying model integrity...")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")

        # Show model classes
        print("\nModel Classes:")
        for idx, name in model.names.items():
            print(f"  {idx}: {name}")

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return model_path

def verify_cuda_availability():
    """Verify CUDA availability for RTX 3080"""
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()

    print(f"\nCUDA available: {cuda_available}")
    print(f"Number of GPUs: {gpu_count}")

    if cuda_available and gpu_count > 0:
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")

        # Set device to GPU
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        print("CUDA not available, using CPU (will be slower)")
        device = torch.device('cpu')

    return device

if __name__ == "__main__":
    device = verify_cuda_availability()
    model_path = download_hockey_model()

    if model_path:
        print(f"\nSetup complete! Model saved at: {model_path}")
        print("You can now use this model in your hockey analytics pipeline.")
    else:
        print("\nSetup failed. Please check the error messages above.")