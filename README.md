# Lung sound classification system

This repository contains the 1) Bi-ResNet for lung sound classification, which is proposed in [this paper](https://ieeexplore.ieee.org/document/8919021). 2) Lung classification system based on a digital stethoscope and an android application, you can find detail information [here](https://ieeexplore.ieee.org/document/8918752)

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Bi-ResNet](#Bi-ResNet)
  * [Pre-processing](#Pre-processing)
  * [Feature](#Feature)
  * [Train](#Train)
  * [Performance](#Performance)
* [Android application](#Android application)
  * [Architecture](#Architecture)
* [Author](#Author)
* [License](#License)
## Bi-ResNet

The architecture of our Bi-ResNet. The input of stft and wavelet are short-time Fourier transform spectrogram and wavelet parameter of one sample![image](https://github.com/mmmmayi/LungSys/blob/master/pic/architecture.png)

## Pre-processing

In order to train the model, you need to download ICBHI 2017 database [here](https://bhichallenge.med.auth.gr/). Each sample provided by this database contains several breath cycles. So you need to clip them according to the start and end time declared officialy. Then you need to divide them into train set and test set. Here we divide them based on official suggestion.

The class to clip samples and divide database are concluded in
```
Bi-Resnet/pre-processing/stft.py
```
named `clip_cycle` and `clip_test` respectively.

## Feature

We implement short-time Fourier transform(stft) and wavelet analysis here to analyze lung sound. you can run 
```
Bi-Resnet/pre-processing/stft.py
Bi-Resnet/pre-processing/wavelet.py
```
respectively and store the spectrogram and wavelet parameter as a picture locally. Then
```
Bi-Resnet/pre-processing/pack.py
```
helps you to store stft spectrogram, wavelet parameters and label into `Bi-Resnet/analysis/pack/wavelet_stft_train.p` and `Bi-Resnet/analysis/pack/wavelet_stft_test.p`

## Train

The model was built using Pytorch, please read detail in 
```
Bi-Resnet/model/bnn.py
```
And it's running commands based on shell:
```
sh run.sh
```

## Performance

Comparison with state-of-the art works:

![image](https://github.com/mmmmayi/LungSys/blob/master/pic/result1.PNG)

Confusion matrix:

![image](https://github.com/mmmmayi/LungSys/blob/master/pic/result2.PNG)
    
## Author

* **Yi Ma** - *Initial work* 

## License

Please cite these papers if you find this project is useful:
```
Y. Ma et al., "LungBRN: A Smart Digital Stethoscope for Detecting Respiratory Disease Using bi-ResNet Deep Learning Algorithm," 2019 IEEE Biomedical Circuits and Systems Conference (BioCAS), Nara, Japan, 2019, pp. 1-4.
 Citation & Abstract
Y. Ma et al., "Live Demo: LungSys - Automatic Digital Stethoscope System For Adventitious Respiratory Sound Detection," 2019 IEEE Biomedical Circuits and Systems Conference (BioCAS), Nara, Japan, 2019, pp. 1-1.
```
