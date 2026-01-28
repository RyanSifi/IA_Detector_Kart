import argparse
import os
import cv2
from ultralytics import YOLO

def process_video(model_path, video_path, output_dir):
    """
    Processes a video to detect karts and saves frames with detections.

    Args:
        model_path (str): Path to the trained YOLOv8 model.
        video_path (str): Path to the input video.
        output_dir (str): Directory to save the frames with detections.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output frames will be saved in: {os.path.abspath(output_dir)}")

    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    saved_frame_count = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has a total of {total_frames} frames.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count > 15582:
            print("\nStopping analysis at 1000 frames as planned.")
            break
        

        print(f"Processing frame {frame_count}/{total_frames}...")

        
        results = model.predict(source=frame, verbose=False)
        
       
        if results and len(results[0].boxes) > 0 and any(box.cls == 0 for box in results[0].boxes):
            saved_frame_count += 1
            output_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"  -> Kart detected! Saved frame to {output_path}")

    cap.release()
    print("\nProcessing complete.")
    print(f"Total frames processed: {frame_count}")
    print(f"Total frames with karts saved: {saved_frame_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video to detect karts and save frames with detections.")
    parser.add_argument(
        "--model",
        type=str,
        default=r"runs/detect/train3/weights/best.pt",
        help="Path to the trained YOLOv8 model weights (.pt file)."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="video_frames_output",
        help="Directory to save the frames where karts are detected."
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at '{args.model}'")
    elif not os.path.exists(args.video):
        print(f"Error: Video file not found at '{args.video}'")
    else:
        process_video(args.model, args.video, args.output_dir)
