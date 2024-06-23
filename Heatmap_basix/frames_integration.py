import cv2
import numpy
import os
import re
from progress.bar import Bar

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(/d+)',text)]

def make_video(image_folder,video_name):
    images = [img for img in os.listdir(image_folder)]
    images.sort(key = natural_keys)
    frame = cv2.imread(os.path.join(image_folder,images[0]))
    h,w,l = frame.shape
    dimensions = (w,h)
    video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'mp4v'),30,dimensions)
    bar= Bar('Video Creation in progress', max = len(images))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder,image)))
        bar.next()
    cv2.destroyAllWindows()
    video.release()

    for file in os.listdir(image_folder):
        os.remove(image_folder+file)