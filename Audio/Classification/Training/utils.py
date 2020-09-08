import os

import pandas as pd
import numpy as np
from scipy.io import wavfile
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def generate_weights(amount_class_array):
    result_weights=amount_class_array/np.sum(amount_class_array)
    result_weights=1./result_weights
    result_weights=result_weights/np.sum(result_weights)
    return result_weights


def how_many_windows_do_i_need(length_sequence, window_size, step):
    """This function calculates how many windows do you need
        with corresponding length of sequence, window_size and
        window_step
        for example, if your sequence length=10, window_size=4 and
        window_step=2 then:
        |_ _ _ _| _ _ _ _ _ _
        _ _ |_ _ _ _| _ _ _ _
        _ _ _ _ |_ _ _ _| _ _
        _ _ _ _ _ _ |_ _ _ _|
        ==> you need 4 windows with this parameters

    :param length_sequence: int, the length of sequence
    :param window_size: int, the length of window
    :param step: int
    :return: int, number of windows needed for this sequence
    """
    start_idx=0
    counter=0
    while True:
        if start_idx+window_size>=length_sequence:
            break
        start_idx+=step
        counter+=1
    if start_idx!=length_sequence:
        counter+=1
    return counter

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          path_to_save=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred).astype('int')]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if path_to_save!='':
        plt.savefig(path_to_save + 'confusion_matrix.png')
    else:
        plt.show()
    return ax

def load_labels(path_to_labels):
    f = open(path_to_labels, 'r')
    original_sample_rate = int(f.readline().split(':')[-1])
    f.close()
    labels=pd.read_csv(path_to_labels, skiprows=1,header=None)
    return labels.values.reshape((-1,)), original_sample_rate

def load_data_wav(path_to_datafile):
    sample_rate, data = wavfile.read(path_to_datafile)
    # if we have 2 channels in audio (stereo)
    if len(data.shape)>1:
        data=data[:,0].reshape((-1,1))
    return data.astype('float32'), sample_rate

def load_data_csv(path_to_datafile):
    data=pd.read_csv(path_to_datafile ,header=None)
    return data.values.astype('float32'), None

def find_the_greatest_class_in_array(array):
    counter_classes=np.unique(array, return_counts=True)
    greatest_class=counter_classes[0][np.argmax(counter_classes[1])]
    return greatest_class

