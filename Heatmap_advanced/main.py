import numpy as np
import time
import cv2
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import pickle

#We are defining the objects that we have to detect using the YOLOv3 in a object file.
labelpath = 'object.names'
labels =[]
# print(labelpath)

#We are trying to open the files and read the contents of the file
f= open(labelpath,'r')
data = f.readlines()

# We are appending the values of the object file into an array "labels"
for d in data:
    labels.append(d[:-1])
# print(labels)

#Checking if a given point lies in a polygon
def point_in_polygon(point, polygon):
    point = Point(point)            #Converts the coordinate into shapely defined point and polygon object
    polygon = Polygon(polygon)
    return polygon.contains(point)  #Returns if the point is in the polygon or not

# We are loading the coordinates of the sections that were defined to be looked for heatmap detection.
sections_pickle_file_path = 'clicked_points.pkl'
with open(sections_pickle_file_path,'rb') as file:
    polygon_points = pickle.load(file)

# We are counting the number of sections that were defined to be looked for heatmap detection.
coordinates_length = len(polygon_points)
n = polygon_points.count(None)
section_count = np.zeros((n+1,1),dtype = 'int')

# print(polygon_points)
# print(n)
# print(section_count)

# We are defining the path of the config and weight files used by the YOLOv3
weight = 'yolov3.weights'
config = 'yolov3.cfg'

# Loading our YOLO detector
nn = cv2.dnn.readNetFromDarknet(config,weight)

#taking the video file as input
video_file = 'input1.mp4'
cap = cv2.VideoCapture(video_file)

# condition to check whether the camera is working or not
if (cap.isOpened()==False):
    print('Error while streaming. Check out the input medium!!!')

#having the count of frames that are there.
lens = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print('Total Frames: ',lens)

# Initialisations
first_init =True
max_p_cnt =0
tot_p_cnt =0
avg_p_cnt = 0
first_iteration_indicator =1
frame_cnt = 0

# Starting loop to work on video frame by frame
while(cap.isOpened()):

    #capturing frames one by one
    ret,frame = cap.read()
    #having the dimensions of the frame
    (height,width) = frame.shape[:2]
    # Creating a blank canvas/array of the same dimension and giving values to the points at which the object is detected in the video.
    if ret ==True:
        blank = np.zeros((height,width),dtype = 'int')
        break

cap = cv2.VideoCapture(video_file)
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret ==True:
        frame_cnt +=1
        layers_name = nn.getLayerNames() #get the output layer names that are there in the YOLO Model
        layers_name = [layers_name[i-1] for i in nn.getUnconnectedOutLayers()]
        # print(layers_name)
        blob = cv2.dnn.blobFromImage(frame,1/255.0,(640,640),swapRB=True,crop=False)
        nn.setInput(blob)
        start= time.time()
        # print(start)
        layerOutputs = nn.forward(layers_name)
        # print(layerOutputs)
        end = time.time()
        # print(end)
        # print(layerOutputs)

        #These are the 3 features of the YOLO model layer that we use.
        boxes = []
        confidences=[]
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # print(confidence)
                if confidence >0.2:
                    # box =detection[0:4]
                    # print(box)
                    box =detection[0:4]*np.array([width,height,width,height]) #having the dimensions of the box which are normalised down are tobe denormalised into original dimensions of the frame.
                    # print(box)
                    (centerX,centerY,width,height) = box.astype('int')
                    topX = int(centerX - (width//2))
                    topY = int(centerY - (height//2))
                    boxes.append([topX,topY,int(width),int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # print(boxes)
        # print(confidences)
        # print(classIDs)
        idx = cv2.dnn.NMSBoxes(boxes,confidences,0.25,0.25)
        # print(idx)
        # print(type(idx))
        # print(len(idx))
        # print(classIDs)
        if len(idx) > 0 :  
            for i in idx.flatten():
                # print(idx.flatten())
                # print(i)
                (x,y) = (boxes[i][0],boxes[i][1])
                (w,h) = (boxes[i][2],boxes[i][3])
                footX = x+w//2
                footY = y+h
                print(classIDs[i])
                print(labels[classIDs[i]])
                print(labels)
                if labels[classIDs[i]-62] == "person":
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),3)
                    cv2.putText(frame,str(int(confidences[i]*100))+'%',(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    none_encounter = 0
                    polygon = []
                    point = (footX,footY)

                    for i in range(coordinates_length):
                        if polygon_points[i] is not None:
                            polygon.append(polygon_points[i])
                            if i == coordinates_length -1 or polygon_points[i+1] is None:
                                check = point_in_polygon(point,polygon)
                                if check:
                                    section_count[none_encounter] +=1
                                polygon =[]
                        else:
                            none_encounter +=1
                            polygon = []

                    polygon = []

                    for i in range(coordinates_length):
                        if polygon_points[i] is not None:
                            polygon.append(polygon_points[i])
                            if (i == coordinates_length -1 or polygon_points[i+1] is None):
                                check = point_in_polygon(point,polygon)
                                if check:
                                    for p in range(-(w//6),(w//6)+1):
                                        for q in range(-(w//6),(w//6)+1):
                                            a = footX +p
                                            b = footY +q
                                            distance = math.sqrt((a-footX)**2 + (b-footY)**2)
                                            if (width -3<a):
                                                continue
                                            if (height -3 <b):
                                                continue
                                            if (point_in_polygon((a,b),polygon)==False):
                                                continue
                                            if ((distance) <=(w//6)):
                                                blank[b,a] +=1
                                    polygon = []
                                else:
                                    none_encounter +=1
                                    polygon =[]
        cv2.imshow('Person Count',frame)
        cv2.waitKey(1)

    else:
        cap.release()
        cv2.destroyAllWindows()
        break

max_val = np.max(blank)

def interpolate(i):
    if i==9:
        R = 227
        G = 35
        B = 27
    elif i==8:
        R = 228
        G = 58
        B = 28
    elif i==7:
        R = 231
        G = 96
        B = 31
    elif i == 6:
        R = 236
        G = 137
        B = 35
    elif i==5:
        R = 241
        G = 178
        B = 39
    elif i==4:
        R = 248
        G = 222
        B = 44
    elif i==3:
        R = 245
        G = 253
        B = 47
    elif i==2:
        R = 210
        G = 252
        B = 44
    elif i==1:
        R = 176
        G = 251
        B = 42
    elif i==0:
        R = 146
        G = 250
        B = 40
    color = [B, G, R]
    return color

def interpolate2(i): # color scheme for the minimum 1% of data recorded for person detection
    if i==9:
        R = 120
        G = 250
        B = 38
    elif i==8:
        R = 103
        G = 249
        B = 37
    elif i==7:
        R = 100
        G = 249
        B = 44
    elif i == 6:
        R = 100
        G = 249
        B = 77
    elif i==5:
        R = 100
        G = 250
        B = 116
    elif i==4:
        R = 101
        G = 250
        B = 157
    elif i==3:
        R = 101
        G = 250
        B = 200
    elif i==2:
        R = 101
        G = 251
        B = 242
    elif i==1:
        R = 89
        G = 219
        B = 251
    elif i==0:
        R = 71
        G = 175
        B = 250
    color = [B, G, R]
    return color

cap = cv2.VideoCapture(video_file)
while True:
    ret,frame = cap.read()
    break

overlay = frame.copy()
height , width, channel = overlay.shape
def remove_less_than_one_percent(array):
    max_val = array[-1]
    threshold = max_val*0.01

    index = 0
    while index <len(array) and array[index] < threshold:
        index +=1
        #as soon as the condition is not getting satisfied, we get the value of index till which we have the values satisfying the threshold values.
        #So we split the array into two parts, one with below threshold values and other one with above threshold values.

    if index >0:
        array_ = array[:index]
        array = array[index:]

        return array, array_
maximum_value_in_data = np.max(blank)
array_1d = blank.flatten()
sorted_array = np.sort(array_1d)
sorted_array,small_values = remove_less_than_one_percent(sorted_array)

def creating_clusters(array):
    unique_array = np.unique(array)

    x = unique_array.reshape(-1,1)

    num_clusters = min(10,len(unique_array))
    
    if num_clusters ==0:
        return [] , [], [],0

    kmeans = KMeans(n_clusters=num_clusters,random_state=0)
    kmeans.fit(x)
    label = kmeans.labels_
    centroids = kmeans.cluster_centers_

    clusters = [[] for _ in range(num_clusters)]
    min_values = [float('inf')] * num_clusters
    max_values = [float('-inf')] * num_clusters

    for i,l in enumerate(label):
        clusters[l].append(unique_array[i])
        if unique_array[i] < min_values[l]:
            min_values[l] = unique_array[i]
        if unique_array[i] > max_values[l]:
            max_values[l] = unique_array[i]
    return clusters, min_values, max_values, num_clusters

def cluster_check(val,min,max,n):
    for i in range(n):
        if(min[i]<=val and val<=max[i]):
            return i

clusters, minval, maxval, num_clus = creating_clusters(sorted_array)
minval_99 = np.sort(minval)
maxval_99 = np.sort(maxval)

for i in range(height):
    for j in range(width):
        if ((0.01* maximum_value_in_data)< blank[i][j]):
            cluster = cluster_check(blank[i][j],minval_99,maxval_99,num_clus)
            color = interpolate(cluster)
            overlay[i,j] = color

clusters, min_vals01, max_vals01,no_of_clusters01 = creating_clusters(small_values)
min_vals01 = np.sort(min_vals01)
max_vals01 = np.sort(max_vals01)

for i in range(height):
    for j in range(width):
        if ((0<blank[i][j])and (blank[i][j]<0.01*maximum_value_in_data)):
            cluster = cluster_check(blank[i][j],min_vals01,max_vals01,no_of_clusters01)
            color = interpolate2(cluster)
            overlay[i,j] = color
cv2.imshow('OverLay',overlay)
output_image = cv2.addWeighted(overlay,0.6,frame,0.3,0)

cv2.imshow('Output',output_image)
cv2.imshow('Frame',frame)
cv2.waitKwy(0)
cv2.destroyAllWindows()
