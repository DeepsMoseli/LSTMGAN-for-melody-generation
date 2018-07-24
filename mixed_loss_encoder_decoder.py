# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 20:52:16 2018

@author: moseli
"""
from numpy.random import seed
seed(1)

from sklearn.model_selection import train_test_split as tts
import logging
import numpy as np


import keras
from keras import backend as k
from sklearn.metrics import log_loss
k.set_learning_phase(1)
from keras.preprocessing.text import Tokenizer
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Dropout,Input,Activation,Add,Concatenate,Lambda,Multiply,Reshape
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

"""-------------------------------------------------------------------------"""
#######################model params####################################
batch_size = 2
num_classes = 1
epochs = 280
dropout=0.2
hidden_units = 128
learning_rate = 0.0001
clip_norm = 2.0
########################################################################
"""--------------------------------------------------------------------------"""


def encoder_decoder():
    
    """____________________________Encoder_____________________________________"""
    
    print('Encoder LSTM layers...')
   
    """___Note__encoder___"""
    Note_encoder_inputs = Input(shape=Note_en_shape)
    Note_encoder = LSTM(hidden_units,return_sequences=True,dropout_U=dropout,
                        dropout_W=dropout,recurrent_initializer='zeros', bias_initializer='ones',return_state=True)
    
    Note_encoder_outputs,Note_state_h,Note_state_c = Note_encoder(Note_encoder_inputs)
    Note_encoder_states = [Note_state_h, Note_state_c]
    
    
    """___type__encoder___"""
    Type_encoder_inputs = Input(shape=type_en_shape)
    Type_encoder = LSTM(hidden_units,return_sequences=True,dropout_U=dropout,
                         dropout_W=dropout,recurrent_initializer='zeros', bias_initializer='ones',return_state=True)
    
    Note_Type_Context = Multiply()([Type_encoder_inputs,Note_encoder_outputs])
    _,Type_state_h,Type_state_c = Type_encoder(Note_Type_Context)
    Type_encoder_states = [Type_state_h,Type_state_c]
    
    
    """____________________________Decoder_____________________________________"""
    
    print('Decoder LSTM layers...')
    
    """___Note__decoder___"""
    Note_decoder_inputs = Input(shape=(None,Note_de_shape[1]))
    Note_decoder = LSTM(hidden_units,dropout_U=dropout,dropout_W=dropout,return_sequences=True,
                        recurrent_initializer='zeros',bias_initializer='ones',return_state=True)
    
    Note_decoder_outputs, _, _ = Note_decoder(Note_decoder_inputs,initial_state=Note_encoder_states)
    Note_decoder_dense = Dense(Note_de_shape[1],activation='softmax',name='note_outputs')
    Note_decoder_outputs = Note_decoder_dense(Note_decoder_outputs)
    
    
    """___Type__decoder___"""
    Type_decoder_inputs = Input(shape=(None,Type_de_shape[1]))
    Type_decoder = LSTM(hidden_units,dropout_U=dropout,dropout_W=dropout,return_sequences=True,
                        recurrent_initializer='zeros',bias_initializer='ones',return_state=True)
    
    Note_Type_Context = Multiply()([Type_decoder_inputs,Note_decoder_outputs])
    
    Type_decoder_outputs,_,_ = Type_decoder(Note_Type_Context,initial_state=Type_encoder_states)
    Type_decoder_dense = Dense(Type_de_shape[1],activation='softmax',name='type_outputs')
    Type_decoder_outputs = Type_decoder_dense(Note_Type_Context)
    
    
    model = Model(inputs=[Note_encoder_inputs,Type_encoder_inputs,Note_decoder_outputs,Type_decoder_outputs],
                  outputs=[Note_decoder_outputs,Type_decoder_outputs])
    print(model.summary())
    
    losses={'note_outputs':'categorical_crossentropy',
            'type_outputs':'categorical_crossentropy'}
    
    model.compile(loss=losses,optimizer='adam',metrics=['accuracy'])
    
    x_note_train,x_note_test,y_note_train,y_note_test=tts(x_note,y_note,test_size=0.2)
    x_type_train,x_type_test,y_type_train,y_type_test=tts(x_type,y_type,test_size=0.2)
    history= model.fit(x=[x_note_train, x_type_train, y_note_train, y_type_train],
              y=[y_note_train,y_type_train],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([x_note_test, x_type_test, y_note_test, y_type_test],
                               [y_note_test,y_type_test]))
    
    """_________________________________Inference Mode______________________________"""
    
    return model,history
    

