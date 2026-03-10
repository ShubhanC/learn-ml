from ultralytics import YOLO
import cv2
import supervision as sv

def live_detection():
    '''Function for the live detection'''

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Initialize the box annotator for visualizing detections
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Main loop for live object detection
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Perform object detection using YOLO
        data = model(frame)[0]
        detection = sv.Detections.from_ultralytics(data)

        # Extract labels and confidence scores for the detected objects
        label = [f"{model.model.names[class_id]} {con:0.2f}"
                 for con, class_id in zip(detection.confidence, detection.class_id)]

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Annotate the frame with bounding boxes and labels
        annotated_frame = box_annotator.annotate(scene=frame, detections=detection)
        frame = label_annotator.annotate(scene=annotated_frame, detections=detection, labels=label)

        # Display the live camera feed with object detection annotations
        cv2.imshow("Live Object Detection", frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

live_detection()