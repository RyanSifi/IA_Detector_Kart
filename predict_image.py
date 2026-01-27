import argparse
import os
from ultralytics import YOLO
from pathlib import Path

def predict(model_path, image_path):
    """
    Runs YOLOv8 prediction on a single image.

    Args:
        model_path (str): Path to the trained YOLOv8 model weights (.pt file).
        image_path (str): Path to the input image.
    """
    # Create output directory if it doesn't exist
    output_dir = "output_predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Load the trained model
    model = YOLO(model_path)

    # Get the base name of the image to create a unique output folder
    image_stem = Path(image_path).stem

    # Run prediction and save the result
    # The result is saved in a folder like 'output_predictions/predict'
    results = model.predict(
        source=image_path,
        save=True,
        project=output_dir,
        name=f"{image_stem}_prediction",
        exist_ok=True # Overwrite previous predictions for the same image
    )

    # The save_dir attribute contains the path to the folder where results are saved
    saved_image_path = results[0].save_dir
    print(f"Prediction complete. Result saved in directory: {saved_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 prediction on a single image.")
    parser.add_argument(
        "--model",
        type=str,
        default=r"runs/detect/train3/weights/best.pt",
        help="Path to the trained YOLOv8 model weights (.pt file)."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image to process."
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at '{args.model}'")
    elif not os.path.exists(args.image):
        print(f"Error: Image file not found at '{args.image}'")
    else:
        predict(args.model, args.image)
