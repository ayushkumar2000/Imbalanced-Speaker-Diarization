# Outliers Team - Speaker Diarization for Imbalanced Class Problem 

Speaker diarization is a task to label audio or video recordings with classes corresponding to speakeridentity, or in short, a task to identify "who spoke when."It involves partitioning an audio stream withmultiple people into homogeneous segments associated with each individual. It is an integral part ofspeech recognition systems and a well-known open problem.
 

## Overview

This project contains:


- Self implemented Spectral clustering algorithm
- Contains trained VAD system
- Text-independent Speaker recognition module based on LSTM and triplet loss.
- GANMM + Spectral Clustering based upsampling
- LSTM Embedding Models on AMI Corpus and VoxConverse 2020 Dataset
- Mainly inspired from research paper: [Speaker Diarization with LSTM](https://arxiv.org/abs/1710.10468]),and [Mixture of GANs for Clustering](https://www.ijcai.org/Proceedings/2018/0423.pdf).



## Prerequisites

- pytorch
- keras
- Tensorflow
- Spectral Clustering
- pyannote.metrics
- pyannote.core
- webrtcvad


(For GANMM )
- python 3.5
- argparse
- pickle
- tensorflow(tested with GPU version) == 1.13.1
- numpy == 1.12.1
- sklearn == 0.18.1


Install all the dependencies with pip command inside colab notebook.
```sh
!pip3 install (above prerequisite)
```





## Data Preprocessing
### A) Extracting MFCC Features: Based on *Data Preprocessing.ipynb* 
For prepairing the training and testing data we extract the MFCC features from the audio files
```sh
y, sr = librosa.load(file_loc,sr = 22050,offset = start+i*audiolen, duration = audiolen)
mfcc = librosa.feature.mfcc(y=y, sr=sr)
```


we have kept the audio lenth = 1
sampling frequency = 22050
mfcc feature shape = 44*20

number of audio files used for training = 24
number of unseen audio files for testing = 3

### B) Generating Custom Imbalanced Data: Based on *Filter Dataset.ipynb*

Here we Generate a custom Imbalanced data by collecting the defined number of MFCC features.
The parameters for customizing data are:
```sh
file = name1
speakers = [1,2,3,4]
samples = [60,100,150,190]    # Set the samples per person for new data
randomize = False             # Option for randomization
```

## Importing Audio Data files

Import the train, test data files in .npy format, with their time-stamps.
Convert the data type to tf.datatype for tensorflow enviornment.
```sh
train_dataset=tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset=train_dataset.batch(32)
```

## Voice Activity Detection
### A) Self trained VAD
Data Generation for the VAD training: Based on *VAD_dataset_generator.ipynb* notebook

Function mel_constructor generates 1 second audio segments Mel spectrograms of dimention 44*128
```sh
y, sr = librosa.load(file_loc,sr = 22050,offset = start+i*audiolen, duration = audiolen)
mel = librosa.feature.melspectrogram(y=y,sr=sr)
```

Function mel_spectrogram_generator parses the transcripts and returns the Mel spectrogram features
```sh
vad_mel_collection_train,vad_labels_train = mel_spectrum_generator(files,training_data)
```

CNN model training for VAD is trained and tested in *VAD_modeling.ipynb* notebook


### B) Using webrtcvad Library
Based on *VAD_library.ipynb* notebook

The audio segments processed individually are based on the following parameters:
```sh
duration = 0.02   #seconds of audio len segments
sr_desired = 16000
```
For audio file *filename*, to get the segments call the function:
```sh
vad(filename)
```
The returned values are of the form:
```sh
y, segments, sr
segments = dict(
    start = starting_time
    end = ending_time
    is_speech = boolian
)
```


## Training

Based on **Main_Pipeline.ipynb** and **CNN_Baseline_Files.ipynb** notebooks.

For training we use tensorflow sequential models.
```sh
model = keras.Sequential()
```
Further we use the following layers:
```sh
layers.LSTM(768)
layers.Dropout(0.01)
layers.Dense(256, activation=None)
layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
```
Finally we train the model with following hyperparameters need to be tuned:
- Optimizer:  SGD, ADAM, etc.
- Learning rate:  0.01, 0.1
- Momentum rate:  0.01, 0.005
- Loss:  Semi triplet loss, semi triplet hard loss, contrastive loss
Now we get our 256-dimension embedding by:
```sh
X_embedding=model.predict(test_dataset)
```


## Clustering

### GANMM+Spectral Clustering (with upsampling)
Refer **GANMM.ipynb** notebook for more informtion.


Use 
```sh
!python "../GANMM-master/main.py" sae_mnist to run GANMM.
```
- This algorithm trained N GANs architetcture for N number of clusters, and also includes pre-training based on Spectral clustering.
- Finally the generated data get stored in fake1.
- To augment fake data into main data to make it balanced run:
```sh
(unique, counts) = np.unique(Y_pred, return_counts=True)
frequencies = np.asarray((unique, counts)).T
for i in range(0,4):
  if(add_num[i]>0):
    X_test = np.append(X_test,fake1[i][:add_num[i]],axis = 0)

```
Then for running Spectral clustering on this data refer to **Clustering Algorithms.ipynb**:

The final cluster is stored in *clusters_sc_g*

OR  

Use the predict() method of class *SpectralClusterer* to perform spectral clustering:
```sh
from spectralcluster import SpectralClusterer
clusterer = SpectralClusterer(
    min_clusters=1,
    max_clusters=10,
    p_percentile=0.98,
    gaussian_blur_sigma=0)
labels = clusterer.predict(X_embedding)
```

Note: GANMM base files have been taken from this [repository](https://github.com/eyounx/GANMM).
### Other clustering methods
To run K means use 
```sh
KmeansMain(X_test,4)
```
To run GMM use
```sh
GMM = GaussianMixture(n_components=4).fit(X_embedding)
y_predict = GMM.predict(X_test)
```

## GANMM on VoxConverse Dataset
## Feature Extraction 
A) Training feature extraction in the file **Feature Extraction Training.ipynb**

Reading the transcript for audio segments
```
log = open(transcript_file,'r')
```
Extracting MFCC features
```
audio,sr = librosa.load(audio_folder+file,sr = 16000,offset = start, duration = audio_len)
mfcc = librosa.feature.mfcc(y = audio, sr = sr, n_mfcc=40)  # Extracting MFCC features
```
B) Testing features extraction in the file **Feature Extraction Testing.ipynb**

Appling Voice Activity Detector to get the estimated audio portion
```
_,segments,_ = vad(audio_folder+file)
for segment in progress(segments):
            if segment['is_speech']==True:
                start = segment['start']
                end = segment['finish']
```
Further the file saves the MFCC features of 1 sec partitions of the audio segments


### LSTM + GANMM model on Voxconverse
Based on the file **LSTM_GANMM_Voxconverse_model.ipynb**

Trains the LSTM model using Tripple hard loss
```
model.compile(
    optimizer=tf.keras.optimizers.SGD( 
        learning_rate=0.003,momentum=0.05),
    loss=tfa.losses.TripletHardLoss())    
```
Getting the spectral clustering and GANMM predictions on embeddings
```
y_pred_spectral = make_spectral_clusters(embeddings_test)
y_pred_ganmm = np.load('./GANMM preds/'+file.split('.')[0]+'.npy')
```



## Testing
Import the pyannote packages.
```sh
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
```
For more info on this refer this [notebook](https://github.com/pyannote/pyannote-metrics/blob/develop/notebooks/pyannote.metrics.diarization.ipynb).

## Contact information
For help or issues, please submit a GitHub issue.
