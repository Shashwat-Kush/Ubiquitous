import cv2
import numpy as np
import pickle  #To store the points array to be used later on
video_path = 'input1.mp4'
cap = cv2.VideoCapture(video_path)
while (cap.isOpened()):
    ret,frame = cap.read()
    height,width = frame.shape[:2]
    print('Frame Height: %d, Frame Width: %d' % (height,width))
    break

frame_copy = frame.copy()
clicked_points = []     #Storing the clicked points
draw_lines = False      #Flag that tells whether to draw lines or not

#MOuse Callback function
def mouse_callback(event,x,y,flags,param):
    global draw_lines

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x,y))
        print(f"Clicked Point: {x},{y}")
        draw_lines = True

#Setting the Mouse Callback Function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image',mouse_callback)

while True:
    cv2.imshow('Image',frame_copy)
    key  = cv2.waitKey(1) & 0xFF

    if key==ord('c'):
        clicked_points.append(None)
        draw_lines = False
    elif key==ord('e'):
        break
    
    if draw_lines and len(clicked_points)>1 :
        for i in range(len(clicked_points)-1):
            pt1 = clicked_points[i]
            pt2 = clicked_points[i+1]
            if pt1  and pt2 :
                cv2.line(frame_copy,pt1,pt2,(0,0,255),3)

print('Clicked Points: ')
print('Section 1:')
t=2
for i in range(len(clicked_points)):
    if clicked_points[i] is None:
        print('\nSection ',t)
        t+=1
    else:
        print(clicked_points[i][0],'\t',clicked_points[i][1])

print(clicked_points)

#Dumping the array into a pickle file
with open ('clicked_points.pkl','wb') as file:
    pickle.dump(clicked_points,file)

cv2.destroyAllWindows()
