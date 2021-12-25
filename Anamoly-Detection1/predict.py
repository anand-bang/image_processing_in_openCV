#############This block has all the static code. 
## Load the pretrained model,sets the context and also defines all other relavant functions

#################Import Tensor Flow Libraries
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, ImageDataGenerator


############## Import os and other libraries
import os
import random
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt


#############This is hardcoded for google drive##########
#os.chdir((r'C:\\Users\\sai\\Downloads\\Anamoly-Detection\\Model_Files\\saved_model\\crack_model'))

############

#########Load the pretrained and saved model
new_model = tf.keras.models.load_model('C:\\Users\\sai\\Downloads\\Anamoly-Detection\\Model_Files\\saved_model\\crack_model')

############Function to plot the image and and also mark it as "Anamoly" or "not a Anamoly"
def plot_activation(img):
  
    pred = new_model.predict(img[np.newaxis,:,:,:])
    pred_class = np.argmax(pred)
    
    if pred_class == 1:
        return "Anamoly"

    else:
        return "No Anamoly"
    
