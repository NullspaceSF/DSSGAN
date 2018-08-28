# DSSGAN
Drum Source Separation via Generative Adversarial Network

## The Problem : Separating Professionally Produced Music

Most audio signals are mixtures of several audio sources (speech, vocals, drums, instruments,noises, ect.) Audio Source Separation is extracting useful spectral, temporal, and spatial features from a given mixture signal, and recovering one or several source signals that minimize interference from other sources, artifacts from music

The majority of research is focused on speech and singing voice separation or Blind Source Separation (BSS) which attempts to extract the contribution of each source of the mixture using no prior knowledge. Typical applications include speech enhancement, music production, and real-time speaker separation for simultaneous translation.The problem of source separation is far from being solved with current state of the art models, and the quality remains insufficient for demanding or production quality applications. Both generative style posterior inference methods and deterministic, supervised posterior approximation models have had some limited success, but both have different constraints that inherently limit improvement of either method on its own:

* For posterior inference models, the correlation between instuments and musical sources in both the time and frequency domain make it difficult to incorporate knowledge of prior distributions of sources, so while the source separation problem could greatly benefit from a Bayesian approach, commonly used generative approaches such as NMF have been constrained by both the required incorrect assumption of time and frequency domain independence, and the complexity required to represent the huge number of spectral bases that are needed to to represent different musical components such as tambre, instruments, synthesized sounds, reverberation, ect. 

* The most effective supervised models have been deep neural networks that directly approximate the posterior distribution by training on labeled multitrack audio data where a ground truth source is available to measure loss.  The big problem with supervised models is the availability of labeled multitrack data.  There are only a few very small datasets available, and they have a big variety of styles so having only a few songs of each particular style and genre leads to a lot of overfitting.  It is also difficult to generate new data in an automated fashion due to the high correlation of sources in professionally produced music.

## The Project: DSSGAN: Drum Source Separation via Generative Adversarial Network 

The problem of audio source separation requires a hybrid approach using a family of models and processing techniques that are tailored for specific inputs and outputs. DSSGAN aims to take advantage of the unique characteristics of percussive sound and the style and features of the source mixture to improve separation performance using the latest state of the art archetectures and various processing and classification methods in order to obtain production or performance quality drum source predicitons as output for use as a back end for music production and DJ software.  

This project addresses Drum Source Separation (DSS) of Professionally Produced Music Recordings using a Semi-Supervised Generative Adversarial Network along with pre and post audio processing techniques to achieve production quality output of unmixed drum audio. I chose this project with the intention that it would be an ongoing project I would approach iteratively, adding new data, models, and archetectures over time until the desired results were acheived. The first phase of the project is training the Adversarial network, evaluating how it performs for percussion and drum separation, and beginning to build a data pipeline for collecting a much larger amount of crowdsourced labeled multitrack training data from the music production and engineering community.


## The Data

The project takes advantage of the open source [stems](https://www.stems-music.com/) multitrack audio format developed by Native Instuments that is built on the mp4 video format that allows DJs to control four different unmixed sources of a stem formatted track independantly.
The the dataset.pkl file should be a dictionary that maps the following strings to the respective dataset partitions:
```
"train_sup" : sample_list
"train_unsup" : [mix_list, acc_list, drums_list]
"train_valid" : sample_list
"train_test" : sample_list
```
A sample_list is a list with each element being a tuple containing three Sample objects. The order for these objects is mixture, accompaniment, drums. You can initialise Sample objects with the constructor of the Sample class found in Sample.py. Each represents an audio signal along with its metadata. This audio should be preferably in .wav format for fast on-the-fly reading, but other formats such as mp3 are also supported.

The entry for "train_unsup" is different since recordings are not paired - instead, this entry is a list containing three lists. These contain mixtures, accompaniment and drums Sample objects respectively. The lists can be of different length. since they are not paired.

You may need to use a batch size that is an even divisor of the lengths of each list to avoid a pesky bug that will throw an error after a few epochs.  The (easiest) fix for me was to use batch size 48 with train_sup of length 192, train_valid and train_test of length 96, and all three unsup lists were length 192. Note you can simply concatenate or split audio files with ffmpeg as needed to adjust the number of audio files without having to add or delete training data.



## Requirements

To run the code, the following Python packages are needed. The GPU version for Tensorflow is needed due to the long running times of this model. You can install them easily using \


```pip install -r requirements.txt``` \



after saving the below list to a text file.\


```python=2.7
tensorflow-gpu>=1.2.0  
sacred>=0.7.0  
audioread>=2.1.5
imageio>=2.2.0
librosa>=0.5.1
lxml>=3.8.0
mir_eval>=0.4
scikits.audiolab>=0.11.0
soundfile>=0.9.0
```\


Note that if using an AWS p2 or p3 instance with the Amazon Deep Learning AMI, you can use the included ```tensorflow_p27``` conda environment.

This does not require the tensorflow-gpu library, as it is an optimized instance that will use the CUDA GPU automatically with standard tensorflow library






## Configuration and hyperparameters

You can configure settings and hyperparameters by modifying the model_config dictionary defined in the beginning of Training.py or using the commandline features of sacred by setting certain values when calling the script via commandline (see Sacred documentation).

Note that alpha and beta (hyperparameters from the paper) as loss weighting parameters are relatively important for good performance, tweaking these might be necessary. These are also editable in the model_config dictionary.

## Training

The code is run by executing

python Training.py

It will train the same separator network first in a purely supervised way, and then using our semi-supervised adversarial approach. Each time, validation performance is measured regularly and early stopping is used, before the final test set performance is evaluated. For the semi-supervised approach, the additional data from dataset["train_unsup"] is used to improve performance.

## Evaluation 

BSS evaluation metrics are computed on the test dataset (SDR, SIR, SAR) - this saves the results in a pickled file along with the name of the dataset.

Logs are written continuously to the logs subfolder, so training can be supervised with Tensorboard. Checkpoint files of the model are created whenever validation performance is tested.


The metrics can be computed with bsseval images v3.0, as described <a href="http://bass-db.gforge.inria.fr/bss_eval/">here</a>. 

## Presentation

Presentation slides, model diagrams, and more information about the entire project can be found [here](./Drum%20Source%20Separation%20via%20Generative%20Adversarial%20Network.pdf). 

Graphic design by [ArtGravity](https://www.artgravitysf.com)


## References

The approach is adapted for drums and percussion from the paper "Adversarial Semi-Supervised Audio Source Separation applied to Singing Voice Extraction" by Daniel Stoller, Sebastian Ewert, and Simon Dixon

More details on the separation method can be found in the following article:

[arXiv:1711.00048v2](https://arxiv.org/abs/1711.00048v2)


## License

MIT License

Copyright (c) 2018 Douglas Reeves

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
