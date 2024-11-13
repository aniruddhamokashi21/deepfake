import dlib
import cv
import os
import re
import json
from pylab import *
# from PIL import image, ImageChops, ImageEnhance 

train_frame_folder = 'dataset\train_sample_videos'

with open(os.path.join(train_frame_folder, 'metadata.json'), 'r') as file:
    data = json.load(file)

list_of_train_data = [f for f in os.listdir(train_frame_folder) if f.endswith('.mp4')]
detector = dlib.get_frontal_face_detector()

for vid in list_of_train_data:
    count = 0
    cap = cv.VideoCapture(os.path.join(train_frame_folder, vid))
    frameRate = cap.get(5)
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if ret != True:
            break
        if frameId % ((int(frameRate)+1)*1) == 0:
            face_rects, scores, idx = detector.run(frame, 0)
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop_img = frame[y1:y2, x1:x2]
                if data[vid]['label'] == 'REAL':
                    cv.imwrite('dataset/real/'+vid.split('.')[0]+'_'+str(count)+'.png', cv.resize(crop_img, (128, 128)))
                elif data[vid]['label'] == 'FAKE':
                    cv.imwrite('dataset/fake/'+vid.split('.')[0]+'_'+str(count)+'.png', cv.resize(crop_img, (128, 128)))
                count+=1