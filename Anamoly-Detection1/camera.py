from flask import Flask, render_template, Response,request
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from predict import plot_activation
from datetime import datetime
import math
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
#from string import count 
from flask import request


SHAPE = (224,224,3)
batch_size = 256
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()
        if ret:
                        
            dim = (SHAPE[0], SHAPE[1])
  
            # resize image
            resized = cv2.resize(frame, dim)
            result = plot_activation(resized)
            #if result == "Anamoly":
             #   cv2.imwrite("C://Users//sai//Downloads//Anamoly-Detection//dump", frame) 
            
            cv2.putText(img=frame, text=result, org=(130, 150), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)

            
        
        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()
