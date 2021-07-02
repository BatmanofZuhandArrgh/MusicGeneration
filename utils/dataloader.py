import os
import sys
sys.path.append("..") 

import warnings
warnings.filterwarnings("ignore", category = UserWarning)

import matplotlib.pyplot as plt
import librosa.display

import glob as glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from utils.utils import FfmpegLoader, LibrosaLoader

NB_AUDIO_SAMPLES = 1321967
SAMPLING_RATE = 44100

class TrainLoader(Dataset):

    def __init__(self, 
        paths = glob.glob('../fma/data/fma_small/*/*'),
        loader_type = 'ffmegpeg',
        ):
        self.paths = paths
        if(loader_type == 'ffmegpeg'):
            self.loader = FfmpegLoader(sampling_rate=SAMPLING_RATE)
        elif(loader_type == 'librosa'):
            self.loader = LibrosaLoader(sampling_rate=SAMPLING_RATE)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = self.loader.load(self.paths[idx])

        return sample, self.paths[idx]


if __name__ == "__main__":
    '''
    # Testing DataLoader
    paths = glob.glob('../fma/data/fma_small/*/*')

    ffmpegloader = FfmpegLoader(sampling_rate=SAMPLING_RATE)
    librosaloader = LibrosaLoader(sampling_rate=SAMPLING_RATE)

    for path in paths:
        print(path)
        sample = librosaloader.load(path)[999:1010]
        sample1 = sample
        plt.figure(figsize=(14, 5))
        plt.plot([x for x in np.arange(len(sample))],sample)
        plt.show()

        sample = ffmpegloader.load(path)[999:1010]
        plt.figure(figsize=(14, 5))
        plt.plot([x for x in np.arange(len(sample))],sample)
        plt.show()

    '''
    trainloader = TrainLoader()
    
    for i, (track,path ) in enumerate(trainloader):
        print(track.shape, path)  

    pass