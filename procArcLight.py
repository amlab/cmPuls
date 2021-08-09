#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 10:22:09 2021

@author: shura
cd C:\RESEARCH\HuySan\data
conda activate
python C:\RESEARCH\HuySan\script\processImage.py "A 10x 120 000007.tif"
forfiles /M *.tif /C "cmd /c if @fsize LEQ 99000000 echo @file"

forfiles /M *.tif /C "cmd /c if @fsize LEQ 99000000 python C:\RESEARCH\HuySan\script\processImage.py @file"
"""

import sys
from skimage import io
import numpy as np
import os
import matplotlib.pyplot as plt
import func

imFile = sys.argv[1]
# file for debug
# imFile = r'C:\RESEARCH\HuySan\SECONDTEST\Rotenone 1 000012.tif'

startFrame = 1
rate = 30#100 # samples/sec
markerMinSize = 5
blurKernel = 5

infLevel = 1
def printInf(msg, lvl):
        if lvl <= infLevel:
            print(msg)
            
printInf("\n\nStart image processing...",1)
printInf("Input file: " + imFile, 1)

im = io.imread(imFile)

recTime = im.shape[0]/rate #sec
fLow = int(0.5*recTime) #sample=freq*recTime
fHigh = int(2*recTime)
w = 0

func.outDir = imFile[:-4]

if not os.path.exists(func.outDir):
    os.makedirs(func.outDir)



#plt.imshow(im[startFrame,:,:]/np.max(im[1,:,:]),cmap='gray')
func.saveImg(im[startFrame,:,:],"startFrame.png")

printInf("Signal/Noise analysis...", 1)

dim1 = im.shape[1]-2*w
dim2 = im.shape[2]-2*w
imp = np.zeros((dim1,dim2))
ima = np.zeros((dim1,dim2))
imf = np.zeros((dim1,dim2))
for x in range(w, dim2):
    if x % 100 == 0: 
        printInf(x, 1)
    for y in range(w, dim1):
        if w == 0:
            d = im[startFrame:, y, x]
        else:    
            d = np.mean(im[startFrame:, y-w:y+w, x-w:x+w], axis=(1,2))
        imp[y,x],imf[y,x],ima[y,x] = func.signalParams(d)
        
        
func.saveImg(imp,"signalMap.png")

#############################################################
import cv2
from scipy.ndimage import gaussian_filter1d
im2 = imp

im2 = cv2.GaussianBlur(im2,(blurKernel, blurKernel),0)
#plt.imshow(im2,cmap='gray')

im2 = cv2.cvtColor(255-(im2*255/np.max(im2)).astype('uint8'), cv2.COLOR_GRAY2BGR)
im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

#plt.imshow(im2,cmap='gray')

ret, thresh = cv2.threshold(im2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#plt.imshow(thresh,cmap='gray')


# Marker labelling
mrkNum, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh)
stats = np.append(stats,np.zeros([len(stats),1]),1)
stats[:,-1]=range(stats.shape[0])
stats_MRKNUM=stats.shape[1]-1
#plt.imshow(markers,cmap='gray')
# rm small markers
markers[np.isin(markers, np.where(stats[:,4] <= markerMinSize))]=0
stats=stats[stats[:,4]>markerMinSize]
#plt.imshow(mrk,cmap='gray')

func.saveImg(markers, "markers.png")

#OK, Add number 
imageWithNumbers = cv2.imread(outDir+r'\\markers.png') 

for i in range(mrkNum):
    cx= int(centroids[i][0])
    cy= int(centroids[i][1])
    cv2.putText(imageWithNumbers, text= str(i+1), org=(cx,cy),
            fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color= (255,0,0),
            thickness=1, lineType=cv2.LINE_AA)
func.saveImg(imageWithNumbers,'markers.png')
###plt.imshow(imageWithNumbers) 

def extractSignal(mrkNum = 1):  
    #m = np.ma.array(im[1,:-w,:-w], mask = markers!=mrkNum )
    #plt.imshow(m,cmap='gray')
    msk=np.broadcast_to(markers!=mrkNum, (im.shape[0]-startFrame, dim1,dim2) )
    #plt.imshow(msk[1,:,:],cmap='gray')
    if w == 0:
        d = np.mean(np.ma.array(im[startFrame:,:,:], mask = msk ),axis=(1,2))
    else:    
        d = np.mean(np.ma.array(im[startFrame:,w:-w,w:-w], mask = msk ),axis=(1,2))
    #plt.plot(d[1:250])
    return(d)

def filtrateSignal(d):    
    base = gaussian_filter1d(d, 100)
    #plt.plot(base)
    d = gaussian_filter1d(d - base, 1.5)
    return(d - np.min(d))
    
#plt.plot(filtrateSignal(extractSignal(mrkList[8])))
# sig=(extractSignal(mrkList[7])**2)[1:]
# wnd = np.hamming(sig.shape[0])
# plt.plot(sig*wnd)
#plt.plot(np.log10(np.abs(calcFFT(extractSignal(mrkList[8])))**2)[3:])


printInf("Export signals...", 1)
stats = np.append(stats,np.zeros([len(stats),1]),1)
stats_QUALITY=stats.shape[1]-1
signals = np.zeros((im.shape[0]-startFrame, stats.shape[0]+1))
signals[:,0] = [i for i in range(im.shape[0]-startFrame)] 
for i in range(stats.shape[0]):
    signals[:,i+1] = extractSignal(stats[i,stats_MRKNUM])
    stats[i, stats_QUALITY] =  func.signalParams(signals[:,i+1],True)[0]
    
for i in range(stats.shape[0]):
    plt.subplot(stats.shape[0],1, i+1)
    plt.axis('off')
    plt.rcParams['font.size'] = '6'
    plt.plot(signals[:,i+1])
    

plt.savefig(outDir+r'\\'+"signals.png", dpi=100)
#plt.show()

with open(outDir+r'\\'+"labels.txt", 'w') as f:
    f.write('left,top,width,height,area,quality\n')
    np.savetxt(f, stats, delimiter=",",fmt='%i' )

    
with open(outDir+r'\\'+"signals.txt", 'w') as f:
    f.write('row\t'+'\t'.join(["Mean"+str(int(stats[e, stats_MRKNUM])) for e in range(stats.shape[0])] ) + '\n')
    np.savetxt(f, signals, delimiter="\t", fmt=(['%i'] + ['%10.2f' for i in range(stats.shape[0])] ) )
    
########################################################### Run R script to calculate parameters 
###printInf("Signal analysys ...", 1)
rScriptName=r'"'+os.path.dirname(__file__)+r'\processSignal.R" '
os.system(r'@"C:\Program Files\R\R-3.6.2\bin\Rscript.exe" '+rScriptName 
          + '"' + outDir + r'\signals.txt" '
          )  




