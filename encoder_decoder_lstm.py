# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 21:10:41 2018

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
from keras.layers import Dense,LSTM,Dropout,Input,Activation,Add,Concatenate
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

"""________________________________________________________________________________"""
#######################model params####################################
batch_size = 2
num_classes = 1
epochs = 300
hidden_units = 129
learning_rate = 0.002
clip_norm = 2.0

#######################################################################
############################Helper Functions###########################
#######################################################################    

x,y = getPairsforencodeco(dataset2,300)

pickleFicle(x,y,"notesOnly_128_263nvar")

#######################if loaded from pickle###########################
data = loadPickle("notes_273var_17_7_2018")   
x= data['x'][:200]
y= data['y'][:200]

#######################################################################

en_shape=np.shape(x[0])
de_shape=np.shape(y[0])

######################################################################


def encoder_decoder():
    print('Encoder_Decoder LSTM...')
   
    """__encoder___"""
    encoder_inputs = Input(shape=en_shape)
    
    encoder_LSTM = LSTM(hidden_units,return_sequences=True,dropout_U=0.2,
                        dropout_W=0.2,recurrent_initializer='zeros', bias_initializer='ones',return_state=True)
    encoder_LSTM2 = LSTM(hidden_units,return_sequences=True,dropout_U=0.2,
                         dropout_W=0.2,recurrent_initializer='zeros', bias_initializer='ones',return_state=True)
    encoder_LSTM3 = LSTM(hidden_units,return_sequences=True,
                         recurrent_initializer='zeros', bias_initializer='ones',return_state=True)
    
    encoder_outputs,_,_ = encoder_LSTM(encoder_inputs)
    encoder_outputs,_,_ = encoder_LSTM2(encoder_outputs)
    encoder_outputs,state_h,state_c = encoder_LSTM3(encoder_outputs)
    
    
    encoder_states = [state_h, state_c]
    
    """____decoder___"""
    decoder_inputs = Input(shape=(None,de_shape[1]))
    decoder_LSTM = LSTM(hidden_units,dropout_U=0.2,dropout_W=0.2,return_sequences=True,
                        recurrent_initializer='zeros',bias_initializer='ones',return_state=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_inputs,initial_state=encoder_states)
    
     
    decoder_dense = Dense(de_shape[1],activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    
    model= Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_outputs)
    print(model.summary())
    
    #rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy',optimizer=rmsprop,metrics=['accuracy'])
    
    x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2)
    history= model.fit(x=[x_train,y_train],
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([x_test,y_test], y_test))
    """_________________inference mode__________________"""
    encoder_model_inf = Model(encoder_inputs,encoder_states)
    
    decoder_state_input_H = Input(shape=(hidden_units,))
    decoder_state_input_C = Input(shape=(hidden_units,)) 
    decoder_state_inputs = [decoder_state_input_H, decoder_state_input_C]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM(decoder_inputs,
                                                                     initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model_inf= Model([decoder_inputs]+decoder_state_inputs,
                         [decoder_outputs]+decoder_states)
    
    scores = model.evaluate([x_test,y_test],y_test, verbose=1)
    
    print('LSTM test scores:', scores)
    print('\007')
    return model,encoder_model_inf,decoder_model_inf,history



"""________________generate song vectors___________"""

def generateSong(sample):
    stop_pred = False
    sample =  np.reshape(sample,(1,en_shape[0],en_shape[1]))
    #get initial h and c values from encoder
    init_state_val = encoder.predict(sample)
    target_seq = np.zeros((1,1,de_shape[1]))
    #target_seq =np.reshape(train_data['summaries'][k][0],(1,1,de_shape[1]))
    generated_song=[]
    while not stop_pred:
        decoder_out,decoder_h,decoder_c= decoder.predict(x=[target_seq]+init_state_val)
        generated_song.append(decoder_out)
        init_state_val= [decoder_h,decoder_c]
        #get most similar word and put in line to be input in next timestep
        #target_seq=np.reshape(model.wv[getWord(decoder_out)[0]],(1,1,emb_size_all))
        target_seq=np.reshape(decoder_out,(1,1,de_shape[1]))
        if len(generated_song)== de_shape[0]:
            stop_pred=True
            break
    return np.array(generated_song).reshape((1,de_shape[0],de_shape[1]))


"""___________________________________________________________________________________"""

trained_model,encoder,decoder,history = encoder_decoder()

"""___________________________________Sample___________________________________________"""

def getNotes(indeX):
    pp=y[indeX]
    sample_song = np.reshape(generateSong(x[indeX]),pp.shape)
    pred_notes=np.argmax(sample_song, axis=-1)
    one_hot_pred=np.zeros(sample_song.shape,dtype=int)
    for mess in range(len(sample_song)):
        one_hot_pred[mess][pred_notes[mess]]=1
    return one_hot_pred


pred=getNotes(9)

pp=y[-1]
log_loss(pp,pred)

ComposeSong(dataset2[6])

print('hello')
