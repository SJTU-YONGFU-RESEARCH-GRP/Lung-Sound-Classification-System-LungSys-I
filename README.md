# Lung sound classification system

This repository contains the 1) Bi-ResNet for lung sound classification, which is proposed in [this paper](https://ieeexplore.ieee.org/document/8919021). 2) Lung classification system based on a digital stethoscope and an android application, you can find detail information [here] (https://ieeexplore.ieee.org/document/8918752)

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Bi-ResNet](#Bi-ResNet)
  * [Pre-processing](#Pre-processing)
  * [Feature extraction](#Feature extraction)
  * [Train](#Train)
  * [Performance](#Performance)
* [Android application](#Android application)
  * [Architecture](#Architecture)

## Bi-ResNet

The architecture of our Bi-ResNet. The input of stft and wavelet are short-time Fourier transform spectrogram and wavelet parameter of one sample![image](https://github.com/mmmmayi/LungSys/blob/master/pic/architecture.png)

## Pre-processing

In order to train the model, you need to download ICBHI 2017 database [here](https://bhichallenge.med.auth.gr/). Each sample provided by this database contains several breath cycles. So you need to clip them according to the start and end time declared officialy. Then you need to divide them into train set and test set. Here we divide them based on official suggestion.

The class to clip samples and divide database are concluded in
```
Bi-Resnet/pre-processing/stft.py
```
named `clip_cycle` and `clip_test` respectively.

## Feature extraction

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

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Release History

* 0.2.1
    * CHANGE: Update docs (module code remains unchanged)
* 0.2.0
    * CHANGE: Remove `setDefaultXYZ()`
    * ADD: Add `init()`
* 0.1.1
    * FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress
    
## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the BSD-3 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
