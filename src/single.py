from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
import argparse
import sys
import os

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# font
font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    #cv 
    parser.add_argument("-face_detector", "--detection_model_path", dest= 'detection_model_path', type=str, help="path of DB", default = './trained_models/detection_models/haarcascade_frontalface_default.xml')
    parser.add_argument("-emotion_model", "--emotion_model_path", dest= 'emotion_model_path', type=str, help="indice of test sets, separated by commas", default = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')
    parser.add_argument("-o", "--output-file", dest= 'outfile', type=str, help="output video file", default = 'output.avi')

    args = parser.parse_args()

    # parameters for loading data and images
    emotion_labels = get_labels('fer2013')
    print(emotion_labels)
    n_classes = len(emotion_labels)
    bar_width = 1000

    # {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    # red = np.asarray((255, 0, 0))
    # green = np.asarray((0, 255, 0))
    # blue = np.asarray((0, 0, 255))
    # yellow = np.asarray((255, 255, 0))
    # cyan = np.asarray((0, 255, 255))
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue =(0, 0, 255)
    yellow = (255, 255, 0)
    cyan = (0, 255, 255)
    colours = [ red, blue, cyan, yellow, blue, cyan, green ]


    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    face_detection = load_detection_model(args.detection_model_path)
    emotion_classifier = load_model(args.emotion_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []

    scalefactor = 2.1

    # starting video streaming
    cv2.namedWindow('window_frame')
    cv2.setWindowProperty('window_frame',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    video_capture = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.outfile,fourcc, 20.0, (int(640*scalefactor),int(480*scalefactor)))
    
    while True:
        ret,bgr_image = video_capture.read()
        if not ret:
            break
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image2  = cv2.resize(rgb_image,(int(640*scalefactor),int(480*scalefactor)), interpolation=cv2.INTER_CUBIC)
        faces = detect_faces(face_detection, gray_image)        

        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            # if emotion_text == 'angry':
            #     color = emotion_probability * np.asarray((255, 0, 0))
            # elif emotion_text == 'sad':
            #     color = emotion_probability * np.asarray((0, 0, 255))
            # elif emotion_text == 'happy':
            #     color = emotion_probability * np.asarray((255, 255, 0))
            # elif emotion_text == 'surprise':
            #     color = emotion_probability * np.asarray((0, 255, 255))
            # else:
            #     color = emotion_probability * np.asarray((0, 255, 0))
            if emotion_probability>0.:
                color = colours[emotion_label_arg]

                # color = color.astype(int)
                # color = color.tolist()
                face_coordinates = (face_coordinates * scalefactor).astype(int)
                # x,y,w,h = face_coordinates
                x = 50
                y = 1000
                w = 500
                
           
                # draw_bounding_box(face_coordinates, rgb_image2, color)
                # draw_text(face_coordinates, rgb_image2, emotion_mode,
                #           color, 0, -45, 1, 1)
                draw_text((x,y), rgb_image2, emotion_text,
                          color, 0, -45, 1, 1)

                cv2.rectangle(rgb_image2, (x,y-5),(x+w,y-15), color, 1)
                cv2.rectangle(rgb_image2, (x,y-5),(x+int(emotion_probability*w),y-15), color, -1)
                
                break

            # bars = w * emotion_prediction[0]
            # # print(bars)
            # o = x
            # for i in range(len(bars)):
            #     cv2.rectangle(rgb_image2, (o,y-5),(o+int(bars[i]),y-15), colours[i], -1)
            #     o += int(bars[i])

            # prob_dist = ""
            # for i in range(n_classes):
            #     prob_dist += emotion_labels.get(i) + ': ' + str('%.2f' %(emotion_prediction[0][i])) + ' '
            
            # cv2.putText(rgb_image2, prob_dist, (50,50), font, 0.7,(255,255,255),2,cv2.LINE_AA)

        bgr_image = cv2.cvtColor(rgb_image2, cv2.COLOR_RGB2BGR)
        out.write(bgr_image)
        fsimg = np.zeros((1080,1920,3),np.uint8)
        h,w,c = bgr_image.shape
        xoff=288
        yoff=36
        fsimg[yoff:yoff+h,xoff:xoff+w,:] = bgr_image[:,:,:]
        cv2.imshow('window_frame', fsimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
