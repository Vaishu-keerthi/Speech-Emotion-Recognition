# Speech Emotion Recognition
Speech is the most natural way of expressing ourselves as humans. It is only natural then to extend this communication medium to computer applications. We define speech emotion recognition (SER) systems as a collection of methodologies that process and classify speech signals to detect the embedded emotions.

This project is an exploration of different audio features and CNN-based architectures for building an effective Speech Emotion Recognition (SER) system. The goal is to improve the accuracy of detecting emotions embedded in speech signals. The repository contains code, notebooks, and detailed explanations of the experiments conducted, including feature extraction techniques, model architectures, training procedures, and evaluation metrics. Contributors interested in speech processing and emotion recognition are welcome to explore and contribute to this project.

**Note:** This README.md file contains an overview of the project, it is recommended to open [notebook](https://github.com/yousefkotp/Speech-Emotion-Recognition/blob/main/main.ipynb) as it contains the code and further explanation for the results.

## Table of Contents
- [Speech Emotion Recognition](#speech-emotion-recognition)
  - [Dataset](#dataset)
    - [Preprocessing](#preprocessing)
    - [Data Augmentation](#data-augmentation)
    - [Data Splitting](#data-splitting)
  - [Features Extraction](#features-extraction)
  - [Building the Models](#building-the-models)
    - [DummyNet](#dummynet)
      - [Model Architecture](#model-architecture)
      - [Using Zero Crossing Rate](#using-zero-crossing-rate)
      - [Using Energy](#using-energy)
      - [Using Zero Crossing Rate and Energy](#using-zero-crossing-rate-and-energy)
      - [Using Mel Spectrogram](#using-mel-spectrogram)
    - [RezoNet](#rezonet)
      - [Model Architecture](#model-architecture-1)
      - [Training](#training)
      - [Classification Report](#classification-report)
    - [ExpoNet](#exponet)
      - [Model Architecture](#model-architecture-2)
      - [Training](#training-1)
      - [Classification Report](#classification-report-1)
  - [Remarks](#remarks)
  - [Results](#results)
  - [Contributors](#contributors)

## Dataset
CREMA (Crowd-sourced Emotional Multimodal Actors Dataset) is a dataset of 7,442 original clips from 91 actors. 7442 samples may be considered a relatively moderate-sized dataset for speech emotion recognition. These clips were from 48 male and 43 female actors between the ages of 20 and 74 coming from a variety of races and ethnicities (African America, Asian, Caucasian, Hispanic, and Unspecified). Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad). The [dataset](https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en) is available on Kaggle.

### Preprocessing
Since the audio files don't have the same length, we will pad them with zeros to make them all the same length to match the length of the largest audio file in the dataset. We will also make sure that all the audio files have the same sampling rate (16 KHz) by resampling them if needed.

### Data Augmentation
Data augmentation is a common technique for increasing the size of a training set by applying random (but realistic) transformations to the audio samples. This helps expose the model to more aspects of the data and generalize better. For this project, we will use the following data augmentation techniques:
- **Noise Injection**
- **Time Shifting**
- **Pitch Shifting**
- **Time Stretching**
- **Volume Scaling**

### Data Splitting
For the data splitting, we will use the following ratios:
- Training Set: 70%
- Testing Set: 30%
- Validation Set: 5% of the Training Set

## Features Extraction
We will process the audio files (wav files) mainly using `librosa` library. We will extract the following features:
- Zero Crossing Rate
- Energy
- Mel Spectrogram

## Building the Models

### DummyNet

#### Model Architecture
- First Convolutional Layer: 1D convolutional layer with 1 input channel, 512 output channels, and a kernel size of 5 and stride of 1.
- First Pooling Layer: Max pooling layer with a kernel size of 5 and stride of 2.
- Second Convolutional Layer: 1D convolutional layer with 512 input channels, 512 output channels, and a kernel size of 5 and stride of 1.
- Second Pooling Layer: Max pooling layer with a kernel size of 5 and stride of 2.
- Third Convolutional Layer: 1D convolutional layer with 512 input channels, 128 output channels, and a kernel size of 5 and stride of 1.
- Third Pooling Layer: Max pooling layer with a kernel size of 5 and stride of 2.
- Flatten: Flattens the input tensor to a 1-dimensional vector.
- First Fully Connected Layer: Fully connected (linear) layer with an input size of `input shape` * 128 and an output size of 256.
- Second Fully Connected Layer: Fully connected (linear) layer with an input size of 256 and an output size of 6.
- Softmax activation is applied to the last layer to produce the output probabilities.

#### Using Zero Crossing Rate

<p align="center">
  <img src="Photos/zcr_report.png?raw=true" alt="Zero Crossing Rate"/>
</p>

#### Using Energy

<p align="center">
  <img src="Photos/energy_report.png?raw=true" alt="Energy"/>
</p>

#### Using Zero Crossing Rate and Energy

<p align="center">
  <img src="Photos/zcr_energy_report.png?raw=true" alt="Zero Crossing Rate and Energy"/>
</p>

#### Using Mel Spectrogram

<p align="center">
  <img src="Photos/melspectogram_report.png?raw=true" alt="Mel Spectrogram"/>
</p>

### RezoNet

#### Model Architecture

<p align="center">
  <img src="Photos/Deep-Net.png?raw=true" alt="RezoNet Model Architecture"/>
</p>

#### Training
For audio processing, we used window size = 512, hop size = 160 and number of mel = 40. We used Adam optimizer with learning rate = 0.00001 and batch size = 16. We trained the model for 86 epochs and saved the best model based on validation accuracy. We used learning rate decay after 50 epochs with decay rate = 0.1. We used L2 regularization with weight decay = 0.01.

After analysis, it turns out that the model is overfitting. We tried to solve this problem by using data augmentation techniques. We used noise injection, time shifting, pitch shifting, time stretching and volume scaling.

The best checkpoint for the model was saved at epoch 5. We will use this checkpoint for testing.

#### Classification Report

<p align="center">
  <img src="Photos/RezoNet_report.png?raw=true" alt="RezoNet Classification Report"/>
</p>

### ExpoNet

#### Model Architecture
The input to the network is expected to be 1D, representing the speech signal which will contain Zero Crossing Rate appended to Energy and appended to 1D flattened Mel Spectrogram.

The network architecture begins with a convolutional layer with 1 input channel, 512 output channels, a kernel size of 5, stride of 1 and same padding. This is followed by a ReLU activation function, batch normalization, and max pooling with a kernel size of 5, stride of 2 and same padding. The shape of the input is updated accordingly.

The process is repeated for four additional convolutional layers with similar configurations but varying number of input and output channels. Each convolutional layer is followed by a ReLU activation, batch normalization, and max pooling layer, with the shape updated after each pooling operation.

After the convolutional layers, the features are flattened using the `nn.Flatten()` module. Then, the flattened features are passed through two fully connected layers with 512 and 6 output units, respectively. The first fully connected layer is followed by a ReLU activation and batch normalization. Finally, the output is passed through a softmax function to obtain the predicted probabilities for each emotion class.

The model can be modified to use `Log Softmax` instead of Softmax for faster computation and practicality reasons. However, using Softmax is not a problem in this case since we are using a GPU to run the model.

#### Training
For audio processing, we used window size = 512, hop size = 160 and number of mel = 40. We used Adam optimizer with learning rate = 0.00001 and batch size = 8. We trained the model for 10 epochs and saved the best model based on validation accuracy. We used L2 regularization with weight decay = 0.001. We have also used data augmentation techniques such as noise injection, time shifting, pitch shifting, time stretching and volume scaling to prevent overfitting.

#### Classification Report

<p align="center">
  <img src="Photos/ExpoNet_report.png?raw=true" alt="ExpoNet Classification Report"/>
</p>

## Remarks
- The README.md file contains an overview of the project, it is recommended to open [notebook](https://github.com/yousefkotp/S

- From all of the previous, it is clear that the best model is ExpoNet. The results can be further improved by using a bigger dataset but we will not be doing that in this project.

- The next step would be to build a two or three layer neural network which will ensemble the previous models and use the output of each model as an input to the neural network. This will help in improving the results even further. However, we don't have the enough time and computational power to do that in this project.

## Results
The results are discussed in further details in the [notebook](https://github.com/yousefkotp/Speech-Emotion-Recognition/blob/main/main.ipynb).

## Contributors

- [Yousef Kotp](https://github.com/yousefkotp)

- [Mohamed Farid](https://github.com/MohamedFarid612)

- [Adham Mohamed](https://github.com/adhammohamed1)
