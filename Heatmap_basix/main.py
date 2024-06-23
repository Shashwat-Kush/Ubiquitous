import cv2
import numpy as np
import re
import os
from frames_integration import make_video
from progress.bar import Bar
import copy

def main():
    cap = cv2.VideoCapture('input.mp4')
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        if not os.path.exists('frames'):
            os.makedirs('frames')
    except OSError:
        print('Error creating directory for frame storage')
    
    bar = Bar('Processing Frames',max = length)
    first_interaction_indicator = 1
    for i in range(0,length):
        ret, frame = cap.read()
        if first_interaction_indicator ==1:
            first_frame = copy.deepcopy(frame)
            height, width = first_frame.shape[:2]
            accum_image = np.zeros((height,width),np.uint8)
            first_interaction_indicator =0 
        else:
            filter = background_subtractor.apply(frame)
            cv2.imwrite('./frame.jpg',frame)
            cv2.imwrite('./diff-bgm-frame.jpg',filter)

            threshold=2
            max_value = 2
            ret, th1 = cv2.threshold(filter,threshold,max_value,cv2.THRESH_BINARY)
            accun_image = cv2.add(accum_image,th1)
            cv2.imwrite('./mask.jpg',accum_image)
            color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_SUMMER)
            video_frame = cv2.addWeighted(frame,0.7,color_image_video,0.7,0)
            color_image = cv2.applyColorMap(accum_image,cv2.COLORMAP_HOT)
            result_overlay = cv2.addWeighted(first_frame,0.7,color_image,0.7,0)
            name =  './frames/frame%d.jpg' % i
            cv2.imwrite(name, result_overlay)

            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        bar.next()
    bar.finish()

    make_video('./frames/','./output.mp4')
    cv2.imwrite('diff-overlay.jpg',result_overlay)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
