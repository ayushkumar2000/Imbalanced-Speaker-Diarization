#!/usr/bin/env python
# coding: utf-8

# # Voice Activity Detertor Using webrtcvad Library
# Here we shall be using the webrtcvad library for detecting the speech and non-speech parts from a audio. The audios will be first converted into appropriate format and sampling rate, and then shall be processed by the library. The output will give us the labels of small segments of audios, flagging them as speech or non-speech.

# In[3]:


import struct           # Importing essential libraries
import webrtcvad
import numpy as np
import librosa
import soundfile as sf
from scipy.io import wavfile
import matplotlib.pyplot as plt


# In[4]:


#Converting an audio file to .wav format with desired sampling rate

def remake_file(filename):
    y, sr = librosa.load(filename,sr=16000)
    sf.write('../temp.wav', y, sr)


# In[6]:


def resegmentation(segments):
    new_segments = []
    i = 0
    while(i<len(segments)):
        j = i
        while j<len(segments) and segments[j]['is_speech']==segments[i]['is_speech']: j+=1

        new_segments.append(dict(
                            is_speech=segments[i]['is_speech'],
                            start = segments[i]['start'],
                            finish = segments[j-1]['finish']))
        i = j

    return new_segments


# In[7]:


def resegmentation_main(segments):
    #print("Number of segments originally: "+str(len(segments)))
    segments = resegmentation(segments)

    while(True):
        del_index = 0
        min_duration = 10000000
        for i in range(len(segments)):
            duration = segments[i]['finish']-segments[i]['start']
            if duration<min_duration:
                del_index = i
                min_duration = duration

        if min_duration>1:
            break

        del segments[del_index]
        segments = resegmentation(segments)

    #print("Number of segments after resegmentation: "+str(len(segments)))
    return segments


# In[8]:


def vad(filename):
    
    duration = 0.02   #in seconds
    frame_samples = int(duration*16000)

    sr, y = wavfile.read(filename)

    vad = webrtcvad.Vad()             # Setting up the library for VAD
    vad.set_mode(3)         # Setting the aggression level of VAD

    y1 = struct.pack("%dh" % len(y), *y)    # Converting signal to bytes format

    segments = []          #For storing the start,finish time and the result of VAD for each frame

    for start in np.arange(0, len(y)-len(y)%sr, frame_samples):
        finish= start + frame_samples

        result = vad.is_speech(y1[2*start:2*finish], sample_rate = sr)  #Predicting

        segments.append(dict(
           is_speech = result,
           start = start/sr,
           finish = finish/sr,))
        
    segments = resegmentation_main(segments)

    return y, segments, sr




