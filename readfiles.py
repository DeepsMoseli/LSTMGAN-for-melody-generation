# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 13:40:24 2018

@author: moseli
"""

import mido
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import pickle as pick
import datetime
import os
import warnings
from sklearn.preprocessing import OneHotEncoder
from mido import Message, MidiFile, MidiTrack, MetaMessage
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.ensemble import RandomForestClassifier as RFC

file_location="C:\\Users\\moseli\\Documents\\Masters of Information technology\\MIT809\\datasets\\midi\\"
ProcData_location ="C:\\Users\\moseli\\Documents\\Masters of Information technology\\MIT809\\code\\saved_data\\"
output_location="C:\\Users\\moseli\\Documents\\Masters of Information technology\\MIT809\\code\\myImplementation\\output\\"


"""________________________________blockout few things________________________________"""
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

#warnings.filterwarnings("ignore")

"""__________________________________HELPER FUNCTIONS___________________________________"""


"""__To load data move adequate number of folders from the clean_midi_source folder to the clean_midi folder__"""

def loadallsongs(dire):
    filenames=[]
    num=1
    count=1
    for dirs,subdr, files in os.walk(dire):
        for fil in files:
            #filenames.append(fil)
            try:
                filenames.append(mido.MidiFile("%s"%(dirs+'\\'+fil)))
                count+=1
            except Exception as e:
                #print("file Number %s: %s"%(num,e))
                pass
            num+=1
    print("Successfully loaded %s of %s Midi files"%(count,num))
    return filenames


#create and write new midi file from existing file
def writefile(filename,midifile,bpm):
    mid = MidiFile(type=2)
    mid.ticks_per_beat=bpm
    for k in midifile:
        track = MidiTrack()
        for j in midifile[k]:
            track.append(j)
        mid.tracks.append(track)
    mid.save('%s%s.mid'%(output_location,filename))
    
"""__generate song for listenning___"""    
def ComposeSong(song):
    mid = MidiFile(type=2)
    mid.ticks_per_beat=120
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    def note_on(val):
        if val<0.7:
            return 'note_off'
        else:
            return 'note_on'
    
    for message in song:
        if message[1]==1:
            track.append(Message(note_on(message[-1]), note=int(message[2]), velocity=int(message[-2]),
                                 time=int(message[0]),channel=1))
    mid.save('%snew_song.mid'%(output_location))



"""___________________________extract all tracks from the channels_______________________"""

def getTracks(song):
    """assuming each instrument is played on a separate channel we can make
        use of Type0 midi file format """
    tracks={}
    nummesseges=0
    for i,track in enumerate(song.tracks):
        #print("track %s: %s"%(i,track.name))
        for k in track:
            if type(k)== mido.messages.messages.Message:
                nummesseges=nummesseges+1
                try:
                    if str(k.channel) not in tracks:
                        tracks[str(k.channel)]=[]
                        if k.type in ['note_on','note_off','program_change','control_change']:
                            tracks[str(k.channel)].append(k)                            
                    else:
                        if k.type in ['note_on','note_off','program_change','control_change']:
                            tracks[str(k.channel)].append(k)
                except Exception as e:
                    next
            else:
                next
    print("%s Messages read"%nummesseges)
    return tracks


"""__________________________________extract training attributes____________________________"""
def CreateAttributes(tracklist):
    tracks={}
    
    for k in tracklist:
        #print("Track: "+k)
        ind=0
        tracks[k]={'type':[],'time':[],'absolutetime':[],'note':[],'velocity':[],
        'control':[],'value':[],'program':[],'channel':[]}
        for p in tracklist[k]:
            try:
                tracks[k]['channel'].append(p.channel)
            except Exception as e:
                tracks[k]['channel'].append('none')
        
            try:
                tracks[k]['type'].append(p.type)
            except Exception as e:
                tracks[k]['type'].append('none')
            
            try:
                tracks[k]['time'].append(p.time)
            except Exception as e:
                tracks[k]['time'].append(0)
            
            try:
                if ind>0:
                    tracks[k]['absolutetime'].append(tracks[k]['absolutetime'][-1]+p.time)
                else:
                     tracks[k]['absolutetime'].append(p.time)
            except Exception as e:
                if ind>0:
                    tracks[k]['absolutetime'].append(tracks[k]['absolutetime'][-1])
                else:
                    tracks[k]['absolutetime'].append(0)
            
            try:
                tracks[k]['note'].append(p.note)
            except Exception as e:
                tracks[k]['note'].append('none')
            
            try:
                tracks[k]['velocity'].append(p.velocity)
            except Exception as e:
                tracks[k]['velocity'].append('none')
            
            try:
                tracks[k]['control'].append(p.control)
            except Exception as e:
                tracks[k]['control'].append('none')
            
            try:
                tracks[k]['value'].append(p.value)
            except Exception as e:
                tracks[k]['value'].append('none')
            
            try:
                tracks[k]['program'].append(p.program)
            except Exception as e:
                tracks[k]['program'].append('none')
    
            ind+=1
    return tracks


def CombineNotes(allnotes):
    n=0
    for k in allnotes:
        if n==0:
            complete=pd.DataFrame(allnotes[k])
        else:
            complete=pd.concat([complete,pd.DataFrame(allnotes[k])])
        n+=1
    
    return complete.sort_values(by=['absolutetime'])
    #return complete

"""
def dummies(frame,feat):
    holder=pd.get_dummies(frame[feat],prefix=feat)
    for k in holder:
         frame[k]=holder[k]
    del frame[feat]
    return frame
"""

def dummies(data,var,n_cat):
    num_channels=n_cat+1
    container={}
    for k in range(1,num_channels):
        container["%s_%s"%(var,k)]=np.zeros(n_cat,dtype=int)
        container["%s_%s"%(var,k)][k-1]=1
    dumms_channel = pd.DataFrame(container)
    hold={}
    for k in dumms_channel:
        hold[k]=[]    
    for k in data[var]:
        for p in hold:
            if p=="%s_%s"%(var,k):
                hold[p].append(1)
            else:
                hold[p].append(0)
    holder2 = pd.DataFrame(hold)
    for k in holder2:
         data[k]=holder2[k]
    del data[var]
    return data
    

def embedding(song,mode):
    try:
        if mode=="notes":
            """----only note info----"""
            #song=song[['velocity']]
            #song=song[['note','channel', 'type', 'velocity']]
            song=song[['note']]
            song=song[song['note']!='none']
            
            """---binarize note_type---"""
            #song['note_on']=list(map(lambda x: 1 if x=='note_on' else 0,song['type']))
            #del song['type']
            
            """---one hot encoding ---"""
            song = dummies(song,'note',128)
            #song = dummies(song,'channel',16)
            #song = dummies(song,'velocity',128)
            return song
        
        elif mode == "all":
            
            """---binarize note_type---"""
            song['note_on']=list(map(lambda x: 1 if x=='note_on' else 0,song['type']))
            del song['type']
            
            """---one hot encoding ---"""
            for k in ['note', 'control', 'program', 'value']:
                song = dummies(song,k)  
            return song
        
        elif mode == 'continuous':
            song=song[['time','channel', 'note', 'type', 'velocity']]
            song=song[song['note']!='none']
            #song=song[song['channel']==1]
            
            """---binarize note_type---"""
            song['note_on']=list(map(lambda x: 1 if x=='note_on' else 0,song['type']))
            del song['type']
            
            return song
            
        else:
            print("Select either NOTES or ALL")
            return "NUll"
    except Exception as e:
        #print(e)
        return "EXCEPT"


def build_dataset(mode):
    """ select mode from (notes,continuous,all)"""
    filenames = loadallsongs(file_location)
    dataset=[]
    for k in range(len(filenames)):
        #song=readfile("%s"%filenames[k])   
        ChannelTracks = getTracks(filenames[k])
        data = CreateAttributes(ChannelTracks)
        completesong = CombineNotes(data)
        completesong = embedding(completesong,mode)
        if type(completesong)!=str:
            dataset.append(completesong)
    print("Dataset build with %s songs"%len(dataset))
    return dataset


def ToNdArray(dataset,size):
    allSongs=[]
    for k in range(len(dataset)):
        if len(dataset[k])<=size:
            next
        else:
            pp = []
            for j in range(size):
                pp.append(dataset[k].iloc[j])
            allSongs.append(pp)
    return np.array(allSongs,dtype=int)

"""__training pairs, (song,true/false label) for GAN training__"""

def TrainingPairs(dataset,latentSize):
    latentSpace = np.random.randint(0,128,size = (int(len(dataset)/2),np.shape(dataset)[1],latentSize))
    
    for j in range(int(np.shape(latentSpace)[0]/2)):
        p=np.random.randint(j+1,len(latentSpace))
        r=np.random.randint(1,10)
        for k in range(len(latentSpace[p])):
            latentSpace[p][k][0]=k*r+p
            latentSpace[p][k][3]=np.random.randint(0,2)
            
    targets = {'real':np.ones(dataset.shape[0]),'fake':np.zeros(latentSpace.shape[0])}
    print("Dimensions of data")
    print(np.shape(dataset),np.shape(latentSpace))
    completedata= np.concatenate((dataset,latentSpace))
    completelabels=np.concatenate((targets["real"],targets["fake"]))
    return completedata,completelabels 


"""__for window training of encoder decoder lstm music generator___"""

def getPairsforencodeco(dataset,window):
    x=[]
    y=[]
    for song in dataset:
        x.append(np.array(song[:window]))
        y.append(np.array(song[window:]))
    x=np.array(x)
    y=np.array(y)
    return x,y    

###########################Pickle################################################
#save processed data in pickle file    
def pickleFicle(data_x,data_y,fileName):
    data = {'x': data_x, 'y': data_y}
    now = datetime.datetime.now()
    date ='%s_%s_%s'%(now.day,now.month,now.year)
    try:
        with open('%s%s_%s.pickle'%(ProcData_location,fileName,date), 'wb') as f:
            pick.dump(data, f)
        status=True
    except Exception as e:
        raise
        status = False
    return "Save pickle status is: %s"%status

#load data dictionary
def loadPickle(fileName):
    with open('%s%s.pickle'%(ProcData_location,fileName), 'rb') as f:
        new_data_variable = pick.load(f)
    return new_data_variable

#################################################################################
def plot_training(history):
    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


"""________________________________________________________________________________________"""

dataset = build_dataset("notes")
dataset2 = ToNdArray(dataset,1000)


dataset2[0][:10]

x_data,y_data=TrainingPairs(dataset2,np.shape(dataset2[0])[2])
"""-----------------------------------------------------------------------"""
