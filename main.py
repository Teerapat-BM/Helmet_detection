from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('best-3.pt')  # Replace with your trained model path

# Initialize video capture (0 for default webcam)
cap = cv2.VideoCapture('test01.mp4')

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(1920,1080))
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform helmet detection
    results = model(frame)

    # Extract bounding boxes and labels from the results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class indices

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = confidences[i]
            class_id = class_ids[i]
            label = model.names[class_id]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Helmet Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
