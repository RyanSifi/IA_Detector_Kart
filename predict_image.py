import argparse
import os
from ultralytics import YOLO
from pathlib import Path

def predict(model_path, image_path):
    output_dir = "output_predictions"
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)

    image_stem = Path(image_path).stem

    results = model.predict(
        source=image_path,
        save=True,
        project=output_dir,
        name=f"{image_stem}_prediction",
        exist_ok=True 
    )

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
