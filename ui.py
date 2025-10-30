from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (for testing)
model = YOLO('yolov8n.pt')

# Run a quick test on a sample image
results = model.predict(source='https://ultralytics.com/images/bus.jpg', show=False)

results[0].show()
print("✅ YOLOv8 is working correctly!")