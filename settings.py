import os

ROOT_FOLDER = '/content/gdrive/MyDrive/SpeakerRecognition'
DATA_PATH = os.path.join(ROOT_FOLDER, 'dataset')
TMP_PATH = os.path.join(ROOT_FOLDER, 'tmp')
WAV_PATH = os.path.join(ROOT_FOLDER, 'data_wav')
WAV_DATA_PATTERN = os.path.join(WAV_PATH, '*/*.wav')

BATCH_SIZE = 4