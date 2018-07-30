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

from keras.utils.np_utils import to_categorical
import keras
from keras import backend as k
from sklearn.metrics import log_loss
k.set_learning_phase(1)
from keras.preprocessing.text import Tokenizer
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Dropout,Input,Activation,Add,Concatenate,\
Lambda,Multiply,Reshape,TimeDistributed,Permute,Flatten,RepeatVector,merge
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

"""-------------------------------------------------------------------------"""
#######################model params####################################
batch_size = 20
num_classes = 1
epochs = 20
dropout=0.2
hidden_units = 128
learning_rate = 0.05
clip_norm = 2.0
########################################################################
"""--------------------------------------------------------------------------"""

###########################################################################


x,y = getPairsforencodeco(dataset2,300)

pickleFicle(x,y,"notes")
pickleFicle(x,y,"msgtype")
pickleFicle(x,y,"velocity")

#######################if loaded from pickle###########################
dataNotes = loadPickle("notes_30_7_2018")   
dataType = loadPickle("msgtype_30_7_2018") 
dataVelocity = loadPickle("velocity_30_7_2018") 


Note_en_shape=np.shape(dataNotes['x'][0])
Note_de_shape=np.shape(dataNotes['y'][0])

Type_en_shape=np.shape(dataType['x'][0])
Type_de_shape=np.shape(dataType['y'][0])

Velocity_en_shape=np.shape(dataVelocity['x'][0])
Velocity_de_shape=np.shape(dataVelocity['y'][0])


##########################################################################
def encoder_decoder():
    
    """____________________________Encoder_____________________________________"""
    
    print('Encoder LSTM layers...')
   
    """___Note__encoder___"""
    Note_encoder_inputs = Input(shape=Note_en_shape)
    Note_encoder_LSTM = LSTM(hidden_units,return_sequences=True,dropout_U=0.2,
                        dropout_W=0.2,recurrent_initializer='zeros', bias_initializer='ones',return_state=True)
    Note_encoder_LSTM2 = LSTM(hidden_units,return_sequences=True,dropout_U=0.2,
                         dropout_W=0.2,recurrent_initializer='zeros', bias_initializer='ones',return_state=True)
    Note_encoder_outputs,_,_ = Note_encoder_LSTM(Note_encoder_inputs)
    Note_encoder_outputs,Note_state_h,Note_state_c = Note_encoder_LSTM2(Note_encoder_outputs)
    Note_encoder_states = [Note_state_h, Note_state_c]
    
    """___type__encoder___"""
    Type_encoder_inputs = Input(shape=Type_en_shape)
    Type_encoder_LSTM = LSTM(hidden_units,return_sequences=True,dropout_U=0.2,
                        dropout_W=0.2,recurrent_initializer='zeros', bias_initializer='ones',return_state=True)
    Type_encoder_outputs,Type_state_h,Type_state_c = Type_encoder_LSTM(Type_encoder_inputs)
    Type_encoder_states = [Type_state_h, Type_state_c]
    
    
    """___Velocity__encoder___"""
    Velocity_encoder_inputs = Input(shape=Velocity_en_shape)
    Velocity_encoder_LSTM = LSTM(hidden_units,return_sequences=True,dropout_U=0.2,
                        dropout_W=0.2,recurrent_initializer='zeros', bias_initializer='ones',return_state=True)
    Velocity_encoder_LSTM2 = LSTM(hidden_units,return_sequences=True,dropout_U=0.2,
                        dropout_W=0.2,recurrent_initializer='zeros', bias_initializer='ones',return_state=True)
    Velocity_encoder_outputs,_, _ = Velocity_encoder_LSTM(Velocity_encoder_inputs)
    Velocity_encoder_outputs,Velocity_state_h,Velocity_state_c = Velocity_encoder_LSTM2(Velocity_encoder_outputs)
    Velocity_encoder_states = [Velocity_state_h, Velocity_state_c]
    
    
    """____________________________Decoder_____________________________________"""
    
    print('Decoder LSTM layers...')
    
    """___Note__decoder___"""
    Note_decoder_inputs = Input(shape=(None,Note_de_shape[1]))
    Note_decoder_LSTM = LSTM(hidden_units,dropout_U=0.2,dropout_W=0.2,return_sequences=True,
                        recurrent_initializer='zeros',bias_initializer='ones',return_state=True)
    Note_decoder_outputs, _, _ = Note_decoder_LSTM(Note_decoder_inputs,initial_state=Note_encoder_states)
    Note_decoder_dense = Dense(Note_de_shape[1],activation='softmax',name="note_outputs")
    Note_decoder_outputs = Note_decoder_dense(Note_decoder_outputs)
    
    
    """___Type__decoder___"""
    Type_decoder_inputs = Input(shape=(None,Type_de_shape[1]))
    Type_decoder_LSTM = LSTM(hidden_units,dropout_U=0.2,dropout_W=0.2,return_sequences=True,
                        recurrent_initializer='zeros',bias_initializer='ones',return_state=True)
    Type_decoder_LSTM2 = LSTM(hidden_units,dropout_U=0.2,dropout_W=0.2,return_sequences=True,
                        recurrent_initializer='zeros',bias_initializer='ones',return_state=True)
    Type_decoder_outputs, _, _ = Type_decoder_LSTM(Type_decoder_inputs,initial_state=Type_encoder_states)
    Note_Type_decoder_outputs = Add()([Type_decoder_outputs,Note_decoder_outputs])
    Type_decoder_outputs, _, _ = Type_decoder_LSTM2(Note_Type_decoder_outputs)
    Type_decoder_dense = Dense(Type_de_shape[1],activation='softmax',name="type_outputs")
    Type_decoder_outputs = Type_decoder_dense(Type_decoder_outputs)
    
    
    """___Velocity__decoder___"""
    Velocity_decoder_inputs = Input(shape=(None,Velocity_de_shape[1]))
    Velocity_decoder_LSTM = LSTM(hidden_units,dropout_U=0.2,dropout_W=0.2,return_sequences=True,
                        recurrent_initializer='zeros',bias_initializer='ones',return_state=True)
    Velocity_decoder_LSTM2 = LSTM(hidden_units,dropout_U=0.2,dropout_W=0.2,return_sequences=True,
                        recurrent_initializer='zeros',bias_initializer='ones',return_state=True)
    Velocity_decoder_outputs, _, _ = Velocity_decoder_LSTM(Velocity_decoder_inputs,initial_state=Velocity_encoder_states)
    Note_velo_decoder_outputs = Add()([Velocity_decoder_outputs,Note_decoder_outputs])
    Velocity_decoder_outputs, _, _ = Velocity_decoder_LSTM2(Note_velo_decoder_outputs)
    Velocity_decoder_dense = Dense(Velocity_de_shape[1],activation='softmax',name="velocity_outputs")
    Velocity_decoder_outputs = Velocity_decoder_dense(Velocity_decoder_outputs)
    

    losses={'note_outputs':'categorical_crossentropy',
            'type_outputs':'categorical_crossentropy',
            'velocity_outputs':'categorical_crossentropy'}
    
    All_inputs = [Note_encoder_inputs,Type_encoder_inputs,Velocity_encoder_inputs,
                  Note_decoder_inputs,Type_decoder_inputs,Velocity_decoder_inputs]
    
    All_outputs = [Note_decoder_outputs,Type_decoder_outputs,Velocity_decoder_outputs] 
    
    model= Model(inputs=All_inputs, outputs=All_outputs)
    rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm)
    model.compile(loss=losses,optimizer=rmsprop,metrics=['categorical_accuracy'])
    
    x_note_train,x_note_test,y_note_train,y_note_test=tts(dataNotes['x'],dataNotes['y'],test_size=0.2)
    x_type_train,x_type_test,y_type_train,y_type_test=tts(dataType['x'],dataType['y'],test_size=0.2)
    x_velocity_train,x_velocity_test,y_velocity_train,y_velocity_test=tts(dataVelocity['x'],dataVelocity['y'],test_size=0.2)
    
    
    history= model.fit(x=[x_note_train,x_type_train,x_velocity_train,
                          y_note_train,y_type_train,y_velocity_train],
              y=[y_note_train,y_type_train,y_velocity_train],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([x_note_test,x_type_test,x_velocity_test,
                                y_note_test,y_type_test,y_velocity_test],
                               [y_note_test,y_type_test,y_velocity_test]))
    
    """_________________________________Inference Mode______________________________"""
    print(model.summary())
    return model,history
    

model,history=encoder_decoder()

history.history.keys()
plot_training(history)
