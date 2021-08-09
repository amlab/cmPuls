# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 16:02:57 2021
usefull functions
@author: A.Merleev
"""
import numpy as np
import math 
from PIL import Image

outDir="."

def saveImg(arr, fname):
    arr = ((arr/np.max(arr))*255).astype('uint8')
    aim = Image.fromarray(arr)
    aim.save(outDir+r'\\'+fname)  
    
def calcFFT(data):
    ft = np.fft.fft(data)/len(data)           # Normalize amplitude
    ft = ft[range(int(len(data)/2))] # Exclude sampling frequency 
    return(ft)    
    
def signalParams(data, normPower=False, fRange=[17,69]):
    ft=calcFFT(data)
    pw = np.abs(ft)**2
    pos=np.argmax(pw[fRange[0]:fRange[1]])
    if (np.max(pw)==0):
        p=0
        f=0
        a=0
    else:    
        if normPower==False:
            p=math.log10(np.max(pw[fRange[0]:fRange[1]])) 
        else:
            p=10*math.log10(np.max(pw[fRange[0]:fRange[1]])/np.max(pw[fRange[1]+1:]))
        f=pos
        a=np.angle(ft,deg=True)[pos]
    return([p,f,a])    
