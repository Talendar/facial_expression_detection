""" Utility functions.

@author Gabriel Nogueira (Talendar)
"""

import os
from py7zr import SevenZipFile
from PIL import Image
import numpy as np

from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output


EXPRESSIONS = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5, "surprise": 6}


class ClearCallback(Callback):
    """ Handles the cleaning of the log during the training of a model. """

    def on_epoch_end(self, epoch, logs=None):
        """ Clears the log. Called when a training epoch ends. """
        clear_output(wait=True)


def load_qider(path, imgsize):
    """ Loads and normalizes the images from the QIDER dataset. """
    data, labels = [], []
    for exp in EXPRESSIONS:
        exp_path = os.path.join(path, exp)
        if not os.path.isdir(exp_path):
            with SevenZipFile(os.path.join(path, exp + ".7z"), 'r') as zip:
                zip.extractall(path)

        for img in os.listdir(exp_path):
            data.append( 
                np.asarray( 
                    Image.open(os.path.join(exp_path, img)).resize(imgsize) 
                ) / 255
            )
            labels.append( EXPRESSIONS[exp] )

    return np.asarray(data).reshape(len(data), imgsize[0], imgsize[1], 1), np.asarray(labels)


def randimg(data, labels, exp_name=None):
    """ Returns a tuple containing a randomly selected image and its label. """
    if exp_name is None:
        i = np.random.randint(0, len(data))
    else:
        i = np.random.choice( [i for i in range(len(labels)) if labels[i] == EXPRESSIONS[exp_name]] )

    return data[i], list(EXPRESSIONS.keys())[list(EXPRESSIONS.values()).index(labels[i])]

