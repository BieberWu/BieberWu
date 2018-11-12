# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:35:28 2018

@author: hp
"""

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

model=InceptionV3()

def predict(model,img):
    img=img.resize((299,299))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    
    preds=model.predict(x)
    return decode_predictions(preds,top=3)[0]

def plot_preds(image,preds):
    plt.figure()
    plt.subplot(211)
    plt.imshow(image)
    
    plt.subplot(212)
    x1=list(reversed(range(len(preds))))
    bar_preds=[pr[2] for pr in preds]
    labels=(pr[1] for pr in preds)
    plt.barh(x1,bar_preds,alpha=0.5)
    
    plt.yticks(x1,labels)
    plt.xlabel('Probability')
    plt.xlim(0,1.1)
    plt.tight_layout()
    plt.show()
    
if __name__=="_main_":
    a=argparse.ArgumentParser()
    a.add_argument("-i")
    arags=a.parse_args()
    
    img=Image.open(args.i)
    preds=predict(model,img)
    print(preds)
    plot_preds(img,preds)