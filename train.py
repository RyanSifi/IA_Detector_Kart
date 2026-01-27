from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')

# Train the model
model.train(data='karting_data.yaml', epochs=10, imgsz=640)
