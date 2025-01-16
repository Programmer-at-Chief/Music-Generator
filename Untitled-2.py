# %%
import time
import os
import cv2
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from bs4 import BeautifulSoup
import string,re
import gensim

from nltk import sent_tokenize
from gensim.utils import simple_preprocess

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential,layers
from tensorflow.keras.optimizers import Adam
import kerastuner as kt
from keras.utils import image_dataset_from_directory,plot_model,pad_sequences
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import * 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import *
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score



# %%
from preprocess import generate_training_sequences,SEQUENCE_LENGTH

import json

json_data = json.load(open('mapping.json','r'))

OUTPUT_UNITS = len(json_data)
LOSS = 'sparse_categorical_crossentropy'
LEARNING_RATE = 0.001
NUM_UNITS = [128,64,16,8]
EPOCHS = 20
BATCH_SIZE = 64 

# %%
def train(output_units = OUTPUT_UNITS,num_units = NUM_UNITS,loss = LOSS,learning_rate= LEARNING_RATE):

    # generate the training sequences
    inputs,targets = generate_training_sequences(SEQUENCE_LENGTH)


    # build the model
    model = build_model(output_units,num_units,loss,learning_rate)
    
    # train the model
    model.fit(inputs,targets,epochs = EPOCHS,batch_size= BATCH_SIZE)

    # output the model
    
    model.save("my_model_gru.keras")


# %%
def build_model(output_units,num_units,loss,learning_rate):

    model = Sequential()
    model.add(Input(shape=(None, output_units)))  # Input layer

    model.add(Bidirectional(GRU(num_units[0], return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(GRU(num_units[1], return_sequences=True))
    model.add(Dropout(0.2))

    # attention = AdditiveAttention()  

    model.add(GRU(num_units[2]))

    model.add(Dense(output_units, activation='softmax')) 

    model.compile(loss=loss,optimizer = Adam(learning_rate=learning_rate),metrics = ['accuracy'])

    model.summary()


    return model


# %%
train()

# %%
print(inputs)


