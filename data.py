import matplotlib.pyplot as plt

from settings import *
from os import path
import os
from pydub import AudioSegment
from scipy.io import wavfile
import librosa
import librosa.display
import numpy as np
import time
import matplotlib as plt
import cv2 as cv


def load_data(audio_dir):
    categories = []
    train_data = []
    test_data = []

    music_folders = os.listdir(audio_dir)
    for g in music_folders:  # g = genres
        path = os.path.join(audio_dir, g)
        if os.path.isdir(path):
            categories.append(g)

            cnt = 0
            files = os.listdir(path)
            for f in files:
                audio_file = os.path.join(path, f)
                if os.path.isfile(audio_file):
                    if cnt % 5 == 2:
                        test_data.append((audio_file, categories.index(g)))
                    else:
                        train_data.append((audio_file, categories.index(g)))
                    cnt += 1

    x_train = np.zeros((len(train_data), 128, 64), dtype='uint8')  # detect spect_size
    y_train = np.zeros((len(train_data)), dtype='uint8')
    x_test = np.zeros((len(test_data), 128, 64), dtype='uint8')
    y_test = np.zeros((len(test_data)), dtype='uint8')

    for i in range(len(train_data)):
        file, g = train_data[i]
        y_train[i] = g
        spec = get_melspectrogram_db(file)
        x_train[i] = spec

    for i in range(len(test_data)):
        file, g = test_data[i]
        y_test[i] = g
        spec = get_melspectrogram_db(file)
        x_test[i] = spec

    rand_train_idx = np.random.RandomState(seed=0).permutation(len(train_data))
    x_train = x_train[rand_train_idx]
    y_train = y_train[rand_train_idx]

    rand_test_idx = np.random.RandomState(seed=0).permutation(len(test_data))
    x_test = x_test[rand_test_idx]
    y_test = y_test[rand_test_idx]

    return categories, train_data, test_data, x_train, y_train, x_test, y_test

def get_melspectrogram_db(file_path, sr=16000, n_fft=512, hop_length=252, n_mels=128):
    try:
        audio,sr = librosa.load(file_path, sr=sr, mono=True)

        if audio.shape[0]<sr:
            audio=np.pad(audio,int(np.ceil((sr-audio.shape[0])/2)),mode='reflect')
        else:
            audio=audio[:sr]

        spec=librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft,
                hop_length=hop_length, n_mels=n_mels)

        return spec

    except Exception as ex:
            print (ex)
            return -1
            pass