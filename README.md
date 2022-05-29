# KAIST EE738 Final Project

## Introduction

The goal of this project is to build a speech recognition system in Korean language.

## Prerequisites

The script is designed to run using the setup given in `Dockerfile`.

## Step 1: Train a baseline model

Part of the code is already given. Fill the rest of the code to train the baseline model. Train for 100 epochs.

### Evaluating the model

The model can be evaluated by using the `--eval` flag. The test set should be defined using `--val_path` and `--val_list` arguments. 

A model trained using the default hyperparameters should give a Character Error Rate (CER) of around 8%.

## Step 2: Improve the model by using a larger dataset

You can use the KSponSpeech dataset to improve the performance of your system. You may not use other sources of data.

You do not need to train for as many epochs, since the training data is much larger than the ClovaCall dataset.

## Step 3: Make other improvements to the model

Here are some ideas that might help to improve the performance of your model.

```
- Change the model architecture
- Change the loss function
- Change the learning rate or use a learning rate scheduler
- Change other hyperparameters
- Use data augmentation
- Use a better decoder
- Train an external language model and perform joint decoding
```

## Step 4: Deploy the model as a REST API

Deploy the model as a REST API. Use the Python notebook to send requests from Colab.

## Submitting your work

When you are done with the project, send me an email with the following in one zip file:

- Your code (`.py`)
- JSON file with the test set output (`.json`)
- A short report summarizing what you have done. Keep it at less than one page at font 14. (`.txt`)