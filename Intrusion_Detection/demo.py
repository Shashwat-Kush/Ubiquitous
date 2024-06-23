import cv2
import numpy as np

# Create a flag to indicate if the line has been drawn
line_drawn = False
start_point, end_point = None, None

# Define a callback function to handle mouse events for drawing the line
def draw_line(event, x, y, flags, param):
    global line_drawn, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        if not line_drawn:
            start_point = (x, y)
            line_drawn = True
        else:
            end_point = (x, y)

# Create a window for the video feed and set the mouse callback function
cv2.namedWindow("Intrusion Detection")
cv2.setMouseCallback("Intrusion Detection", draw_line)

# Define the video source (0 for webcam or specify a video file)
video_source = 'input1_.mp4'  # Change this to your video source

# Load the video source
cap = cv2.VideoCapture(video_source)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Get the frame dimensions
    height, width = frame.shape[:2]

    if line_drawn:
        # Draw the line
        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

        # Check if both start_point and end_point are defined
        if start_point is not None and end_point is not None:
            # Define the inspection area
            line_y = start_point[1]
            roi_y1 = min(start_point[1], end_point[1])
            roi_y2 = max(start_point[1], end_point[1])

            # Convert the frame to grayscale for background subtraction
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform background subtraction (you can use more advanced methods here)
            if 'background' not in locals():
                background = gray_frame
                continue

            diff_frame = cv2.absdiff(background, gray_frame)

            # Apply a threshold to detect motion
            threshold = 30
            _, threshold_frame = cv2.threshold(diff_frame, threshold, 255, cv2.THRESH_BINARY)

            # Find contours of the detected objects
            contours, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check for objects crossing the line or entering the ROI
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                object_center_x = x + w // 2
                object_center_y = y + h // 2

                # Check if the object crosses the line
                if y < line_y and (y + h) > line_y:
                    cv2.putText(frame, "Intrusion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Check if the object enters the ROI
                if roi_y1 < object_center_y < roi_y2:
                    cv2.putText(frame, "Intrusion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Intrusion Detection', frame)

    # Press 'q' to exit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
