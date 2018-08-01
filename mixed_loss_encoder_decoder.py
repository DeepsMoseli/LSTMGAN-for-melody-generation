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
batch_size = 15
num_classes = 1
epochs = 250
dropout=0.2
hidden_units = 128
learning_rate = 0.0005
clip_norm = 2.0
########################################################################
"""--------------------------------------------------------------------------"""

###########################################################################


x,y = getPairsforencodeco(dataset2,300)

pickleFicle(x,y,"notes")
pickleFicle(x,y,"msgtype")
pickleFicle(x,y,"velocity")
pickleFicle(x,y,"time")

#######################if loaded from pickle###########################
dataNotes = loadPickle("notes_31_7_2018")   
dataType = loadPickle("msgtype_31_7_2018") 
dataVelocity = loadPickle("velocity_31_7_2018") 
dataTime = loadPickle("time_31_7_2018") 


Note_en_shape=np.shape(dataNotes['x'][0])
Note_de_shape=np.shape(dataNotes['y'][0])

Type_en_shape=np.shape(dataType['x'][0])
Type_de_shape=np.shape(dataType['y'][0])

Velocity_en_shape=np.shape(dataVelocity['x'][0])
Velocity_de_shape=np.shape(dataVelocity['y'][0])

Time_en_shape=np.shape(dataTime['x'][0])
Time_de_shape=np.shape(dataTime['y'][0])

##########################################################################

def attention(independent_decoder_output,dependent_decoder_output):
    attention = TimeDistributed(Dense(1, activation = 'tanh'))(independent_decoder_output)
    attention = Flatten()(attention)
    attention = Multiply()([attention, dependent_decoder_output])
    attention = Activation('softmax')(attention)
    attention = Permute([2, 1])(attention)
    return attention
 

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
    
    """___Time__encoder___"""
    Time_encoder_inputs = Input(shape=Time_en_shape)
    Time_encoder_LSTM = LSTM(hidden_units,return_sequences=True,dropout_U=0.2,
                        dropout_W=0.2,recurrent_initializer='zeros', bias_initializer='ones',return_state=True)
    Time_encoder_LSTM2 = LSTM(hidden_units,return_sequences=True,dropout_U=0.2,
                        dropout_W=0.2,recurrent_initializer='zeros', bias_initializer='ones',return_state=True)
    Time_encoder_outputs,_, _ = Time_encoder_LSTM(Time_encoder_inputs)
    Time_encoder_outputs,Time_state_h,Time_state_c = Time_encoder_LSTM2(Time_encoder_outputs)
    Time_encoder_states = [Time_state_h, Time_state_c]
    
    
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
    #Note_Type_decoder_outputs = Add()([Type_decoder_outputs,Note_decoder_outputs])
    Type_decoder_outputs, _, _ = Type_decoder_LSTM2(Type_decoder_outputs)
    Type_decoder_dense = Dense(Type_de_shape[1],activation='softmax',name="type_outputs")
    Type_decoder_outputs = Type_decoder_dense(Type_decoder_outputs)
    
    
    """___Velocity__decoder___"""
    Velocity_decoder_inputs = Input(shape=(None,Velocity_de_shape[1]))
    Velocity_decoder_LSTM = LSTM(hidden_units,dropout_U=0.2,dropout_W=0.2,return_sequences=True,
                        recurrent_initializer='zeros',bias_initializer='ones',return_state=True)
    Velocity_decoder_LSTM2 = LSTM(hidden_units,dropout_U=0.2,dropout_W=0.2,return_sequences=True,
                        recurrent_initializer='zeros',bias_initializer='ones',return_state=True)
    Velocity_decoder_outputs, _, _ = Velocity_decoder_LSTM(Velocity_decoder_inputs,initial_state=Velocity_encoder_states)
    #Note_velo_decoder_outputs = Add()([Velocity_decoder_outputs,Note_decoder_outputs])
    Velocity_decoder_outputs, _, _ = Velocity_decoder_LSTM2(Velocity_decoder_outputs)
    Velocity_decoder_dense = Dense(Velocity_de_shape[1],activation='softmax',name="velocity_outputs")
    Velocity_decoder_outputs = Velocity_decoder_dense(Velocity_decoder_outputs)
    
    
    """___Time__decoder___"""
    Time_decoder_inputs = Input(shape=(None,Time_de_shape[1]))
    Time_decoder_LSTM = LSTM(hidden_units,dropout_U=0.2,dropout_W=0.2,return_sequences=True,
                        recurrent_initializer='zeros',bias_initializer='ones',return_state=True)
    Time_decoder_LSTM2 = LSTM(hidden_units,dropout_U=0.2,dropout_W=0.2,return_sequences=True,
                        recurrent_initializer='zeros',bias_initializer='ones',return_state=True)
    Time_decoder_outputs, _, _ = Time_decoder_LSTM(Time_decoder_inputs,initial_state=Time_encoder_states)
    #Note_velo_decoder_outputs = Add()([Time_decoder_outputs,Note_decoder_outputs])
    Time_decoder_outputs, _, _ = Time_decoder_LSTM2(Time_decoder_outputs)
    Time_decoder_dense = Dense(Time_de_shape[1],activation='relu',name="time_outputs")
    Time_decoder_outputs = Time_decoder_dense(Time_decoder_outputs)
    

    losses={'note_outputs':'categorical_crossentropy',
            'type_outputs':'categorical_crossentropy',
            'velocity_outputs':'categorical_crossentropy',
            'time_outputs':'mse'}
    
    All_inputs = [Note_encoder_inputs,Type_encoder_inputs,Velocity_encoder_inputs,Time_encoder_inputs,
                  Note_decoder_inputs,Type_decoder_inputs,Velocity_decoder_inputs,Time_decoder_inputs]
    All_outputs = [Note_decoder_outputs,Type_decoder_outputs,Velocity_decoder_outputs,Time_decoder_outputs] 
    
    
    All_metrics = ['accuracy']
    
    model= Model(inputs=All_inputs, outputs=All_outputs)
    rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm)
    model.compile(loss=losses,optimizer=rmsprop,metrics=All_metrics)
    
    x_note_train,x_note_test,y_note_train,y_note_test=tts(dataNotes['x'],dataNotes['y'],test_size=0.2)
    x_type_train,x_type_test,y_type_train,y_type_test=tts(dataType['x'],dataType['y'],test_size=0.2)
    x_velocity_train,x_velocity_test,y_velocity_train,y_velocity_test=tts(dataVelocity['x'],dataVelocity['y'],test_size=0.2)
    x_time_train,x_time_test,y_time_train,y_time_test=tts(dataTime['x'],dataTime['y'],test_size=0.2)
    
    history= model.fit(x=[x_note_train,x_type_train,x_velocity_train,x_time_train,
                          y_note_train,y_type_train,y_velocity_train,y_time_train],
              y=[y_note_train,y_type_train,y_velocity_train,y_time_train],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([x_note_test,x_type_test,x_velocity_test,x_time_test,
                                y_note_test,y_type_test,y_velocity_test,y_time_test],
                               [y_note_test,y_type_test,y_velocity_test,y_time_test]))
    
    """####################################Inference Mode##############################################"""
    Note_encoder_model_inf = Model(Note_encoder_inputs,Note_encoder_states)
    Type_encoder_model_inf = Model(Type_encoder_inputs,Type_encoder_states)
    Velocity_encoder_model_inf = Model(Velocity_encoder_inputs,Velocity_encoder_states)
    Time_encoder_model_inf = Model(Time_encoder_inputs,Time_encoder_states)
    
    """________________________________Note_______________________________________"""
    Note_decoder_state_input_H = Input(shape=(hidden_units,))
    Note_decoder_state_input_C = Input(shape=(hidden_units,)) 
    Note_decoder_state_inputs = [Note_decoder_state_input_H, Note_decoder_state_input_C]
    Note_decoder_outputs, Note_decoder_state_h, Note_decoder_state_c = Note_decoder_LSTM(Note_decoder_inputs,
                                                                     initial_state=Note_decoder_state_inputs)
    Note_decoder_states = [Note_decoder_state_h, Note_decoder_state_c]
    Note_decoder_outputs = Note_decoder_dense(Note_decoder_outputs)
    
    """_______________________________Velocity____________________________________"""
    Velocity_decoder_state_input_H = Input(shape=(hidden_units,))
    Velocity_decoder_state_input_C = Input(shape=(hidden_units,)) 
    Velocity_decoder_state_inputs = [Velocity_decoder_state_input_H, Velocity_decoder_state_input_C]
    Velocity_decoder_outputs, _, _ = Velocity_decoder_LSTM(Velocity_decoder_inputs,
                                                                     initial_state=Velocity_decoder_state_inputs)
    #Note_velo_decoder_outputs = Add()([Velocity_decoder_outputs,Note_decoder_outputs])
    Velocity_decoder_outputs, Velocity_decoder_state_h, Velocity_decoder_state_c = Velocity_decoder_LSTM2(Velocity_decoder_outputs)
    
    Velocity_decoder_states = [Velocity_decoder_state_h, Velocity_decoder_state_c]
    Velocity_decoder_outputs = Velocity_decoder_dense(Velocity_decoder_outputs)
    
    """___________________________________Type____________________________________"""
    Type_decoder_state_input_H = Input(shape=(hidden_units,))
    Type_decoder_state_input_C = Input(shape=(hidden_units,)) 
    Type_decoder_state_inputs = [Type_decoder_state_input_H, Type_decoder_state_input_C]
    Type_decoder_outputs, _, _ = Type_decoder_LSTM(Type_decoder_inputs,
                                                                     initial_state=Type_decoder_state_inputs)
    #Note_velo_decoder_outputs = Add()([Type_decoder_outputs,Note_decoder_outputs])
    Type_decoder_outputs, Type_decoder_state_h, Type_decoder_state_c = Type_decoder_LSTM2(Type_decoder_outputs)
    
    Type_decoder_states = [Type_decoder_state_h, Type_decoder_state_c]
    Type_decoder_outputs = Type_decoder_dense(Type_decoder_outputs)
    
    
    """___________________________________Time____________________________________"""
    Time_decoder_state_input_H = Input(shape=(hidden_units,))
    Time_decoder_state_input_C = Input(shape=(hidden_units,)) 
    Time_decoder_state_inputs = [Time_decoder_state_input_H, Time_decoder_state_input_C]
    Time_decoder_outputs, _, _ = Time_decoder_LSTM(Time_decoder_inputs,
                                                                     initial_state=Time_decoder_state_inputs)
    #Note_velo_decoder_outputs = Add()([Time_decoder_outputs,Note_decoder_outputs])
    Time_decoder_outputs, Time_decoder_state_h, Time_decoder_state_c = Time_decoder_LSTM2(Time_decoder_outputs)
    
    Time_decoder_states = [Time_decoder_state_h, Time_decoder_state_c]
    Time_decoder_outputs = Time_decoder_dense(Time_decoder_outputs)
    
    
    """_________________________________Deco models___________________________________________"""
    Note_decoder_model_inf= Model([Note_decoder_inputs]+Note_decoder_state_inputs,
                         [Note_decoder_outputs]+Note_decoder_states)
    
    Type_decoder_model_inf= Model([Type_decoder_inputs]+Type_decoder_state_inputs,
                         [Type_decoder_outputs]+Type_decoder_states)
    
    Velocity_decoder_model_inf= Model([Velocity_decoder_inputs]+Velocity_decoder_state_inputs,
                         [Velocity_decoder_outputs]+Velocity_decoder_states)
    
    Time_decoder_model_inf= Model([Time_decoder_inputs]+Time_decoder_state_inputs,
                         [Time_decoder_outputs]+Time_decoder_states)
    

    """"__________________________________________________________________________________"""
    
    scores = model.evaluate([x_note_test,x_type_test,x_velocity_test,x_time_test,
                             y_note_test,y_type_test,y_velocity_test,y_time_test],
                            [y_note_test,y_type_test,y_velocity_test,y_time_test],
                            verbose=1)
    
    individual_models={'note':[Note_encoder_model_inf,Note_decoder_model_inf],
                       'type':[Type_encoder_model_inf,Type_decoder_model_inf],
                       'velocity':[Velocity_encoder_model_inf,Velocity_decoder_model_inf],
                       'time':[Time_encoder_model_inf,Time_decoder_model_inf]}
    
    print(model.summary())
    return model,individual_models,history


##################################################################################
def generateSong(sample,en_shape,de_shape,encoder,decoder):
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


def getNotes(sample):
    pred_notes=np.argmax(sample, axis=-1)
    return pred_notes
  

def composeSong(index):
    sample_song_Notes = np.reshape(generateSong(dataNotes['x'][index],Note_en_shape,Note_de_shape,
                                          AllModels['note'][0],AllModels['note'][1]),dataNotes['y'][index].shape)
    
    sample_song_Velocity = np.reshape(generateSong(dataVelocity['x'][index],Velocity_en_shape,Velocity_de_shape,
                                          AllModels['velocity'][0],AllModels['velocity'][1]),dataVelocity['y'][index].shape)
    
    sample_song_Type = np.reshape(generateSong(dataType['x'][index],Type_en_shape,Type_de_shape,
                                          AllModels['type'][0],AllModels['type'][1]),dataType['y'][index].shape)
    
    sample_song_Time = np.reshape(generateSong(dataTime['x'][index],Time_en_shape,Time_de_shape,
                                          AllModels['time'][0],AllModels['time'][1]),dataTime['y'][index].shape)
    completesong={}
    completesong['notes']=list(getNotes(sample_song_Notes))
    completesong['velocity']=list(getNotes(sample_song_Velocity))
    completesong['type']=list(map(lambda x:'note_on' if x==0 else 'note_off', getNotes(sample_song_Type)))
    completesong['time'] = list(map(int,sample_song_Time))
    
    mid = MidiFile(type=2)
    mid.ticks_per_beat=120
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)    
    for k in range(len(completesong['time'])):
        track.append(Message(completesong['type'][k],
                             note=completesong['notes'][k],
                             velocity=completesong['velocity'][k],
                             time=int(completesong['time'][k]),
                             channel=1))
    mid.save('%snew_song_%s.mid'%(output_location,index))



model,AllModels,history = encoder_decoder()
history.history.keys()
plot_training(history)
composeSong(10)    
