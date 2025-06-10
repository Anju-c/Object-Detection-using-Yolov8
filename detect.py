from ultralytics import YOLO
import cv2

# Load YOLOv8 Nano Model (fastest)
model = YOLO('yolov8n.pt')

# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Perform detection
    results = model.predict(source=frame, conf=0.5, show=True)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
