# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:15:20 2019

@author: xb
"""

import joblib
import os
from PIL import Image
import numpy as np


def pack(dir_stft,dir_wavelet,label):       
    feature_stft_list=[]
    feature_wavelet_list=[]
    label_list=[] 
    for file in os.listdir(dir_stft):
        I_stft = Image.open(dir_stft+file).convert('L')
        I_wavelet = Image.open(dir_wavelet+file).convert('L')
        I_stft = np.array(I_stft)
        I_wavelet = np.array(I_wavelet)

        feature_wavelet_list.append(I_wavelet)
        feature_stft_list.append(I_stft)
        label_list.append(label)
    return feature_stft_list,feature_wavelet_list,label_list
    

    
if __name__ == '__main__':


    stft0,wavelet0,label0 = pack('../analysis/stft/test/zero/','../analysis/wavelet/test/zero/',0)
    stft1,wavelet1,label1 = pack('../analysis/stft/test/one/','../analysis/wavelet/test/one/',1)
    stft2,wavelet2,label2 = pack('../analysis/stft/test/two/','../analysis/wavelet/test/two/',2)
    stft3,wavelet3,label3 = pack('../analysis/stft/test/three/','../analysis/wavelet/test/three/',3)
    stft = stft0+stft1+stft2+stft3
    wavelet = wavelet0+wavelet1+wavelet2+wavelet3
    label = label0+label1+label2+label3
#    joblib.dump((stft,wavelet, label), open('../analysis/pack/wavelet_stft_test.p', 'wb'))