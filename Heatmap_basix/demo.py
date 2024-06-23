import cv2
import numpy as np
import os
# Load the video
# video_path = 'input1.mp4'
video_path = 'input.mp4'
# video_path =0
cap = cv2.VideoCapture(video_path)
print("file exists?", os.path.exists(video_path))
# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=False)

# Create an empty video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
output_path = 'output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def generate_heat_map(frame, motion_mask):
    # Convert the motion mask to a heatmap
    heatmap = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)
    
    # Overlay the heatmap on the original frame
    overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
    
    return overlay

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Apply background subtraction to detect motion
    motion_mask = bg_subtractor.apply(frame)
    
    # Generate the heat map for the detected motion regions
    heatmap_frame = generate_heat_map(frame, motion_mask)
    
    # Write the frame with the heat map to the output video
    out.write(heatmap_frame)
    
    # Display the frame with the heat map (optional)
    cv2.imshow('Motion Heat Map', heatmap_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects
cap.release()
out.release()
cv2.destroyAllWindows()
