import cv2
import numpy 
import re 
import os     
from os.path import isfile, join

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn)if isfile(join(pathIn,f))] #listdir function lists all the files in the directory and the isfile function checks whether the selected thing is a file or not. If it is a file, it returns True ,otherwise False.
    def get_digits(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [get_digits(c) for c in re.split(r'(/d+)',text)]
    files.sort(key=natural_keys)
    dimensions = (0,0)
    for i in range(len(files)):
        filename = pathIn + files[i]
        img = cv2.imread(filename)
        h,w,l = img.shape
        dimensions = (w,h)
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'),fps,dimensions)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()
    
def main():
    pathIn='./frames/'
    pathOut = 'video.mp4'
    fps = 30.0
    convert_frames_to_video(pathIn,pathOut,fps)

if __name__=="__main__":
    main()