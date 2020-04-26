#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:40:07 2019

@author: mayi
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import pywt
import math
import os
import librosa.display as display
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def read_txt(dir):

    a = np.loadtxt(dir)
    return a  

def wavelet(sig):
    cA, out = pywt.dwt(sig, 'db8')
    cA, out = pywt.dwt(cA, 'db8')
    cA, out = pywt.dwt(cA, 'db8')
    A = cA
    
    for i in range(6):
        cA, cD = pywt.dwt(A, 'db8')
        A = cA
        out = np.hstack((out,cD))

    out = np.hstack((out,A))
        
    return out

def reshape(matrix):
    num = matrix.shape[0]
    length = math.ceil(np.sqrt(num))
    zero = np.zeros([np.square(length)-num,])
    matrix = np.concatenate((matrix,zero))
    out = matrix.reshape((length,length))
    return out

def Normalization(x):
    x = x.astype(float)
    max_x = max(x)
    min_x = min(x)
    for i in range(len(x)):
        
        x[i] = float(x[i]-min_x)/(max_x-min_x)
           
    return x
   
def save_pic(wav_dir,save_dir):
    txt_name = ''
    txt_dir ='../data/ICBHI/'
    for file in os.listdir(wav_dir):
        num = file[-5]
        if file[:22]!=txt_name[:-4]:
            txt_name = file[:22]+'.txt'
            array = np.loadtxt(txt_dir+txt_name)
            label = array[:,2:4]
            
        fs,sig= wav.read(wav_dir+'/'+file)
        sig = Normalization(sig)
        if fs>4000:
            sig = butter_bandpass_filter(sig, 1, 4000, fs, order=3)
            
        wave = wavelet(sig)
        xmax=max(wave)
        xmin=min(wave)
        wave=(255-0)*(wave-xmin)/(xmax-xmin)+0       
        wave = reshape(wave)
        display.specshow(wave)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        
        crackles = label[int(num),0]
        wheezes = label[int(num),1]
        if crackles==0 and wheezes==0:
            plt.savefig(save_dir+'zero/'+file[:24]+'png', cmap='Greys_r')
        elif crackles==1 and wheezes==0:
            plt.savefig(save_dir+'one/'+file[:24]+'png', cmap='Greys_r')
        elif crackles==0 and wheezes==1:
            plt.savefig(save_dir+'two/'+file[:24]+'png', cmap='Greys_r')
        else:
            plt.savefig(save_dir+'three/'+file[:24]+'png', cmap='Greys_r')
        plt.close()
            
if __name__ == '__main__':

    save_pic('../data/test_set','../analysis/wavelet/test/')