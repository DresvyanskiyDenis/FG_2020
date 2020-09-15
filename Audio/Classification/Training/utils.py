import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from Audio.Classification.Preprocessing.labels_utils_regression import transform_probabilities_to_original_sample_rate
from Audio.Classification.Training.Database_instance import Database_instance
from Audio.Classification.Training.Generator_audio import predict_data_with_the_model


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
                          mode='show',
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
    if mode=='show':
        plt.show()
    elif mode=='save':
        plt.savefig(path_to_save + 'confusion_matrix.png')

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

def generate_test_predictions(path_to_data, labels_filename, model, model_output_sample_rate, path_to_video, window_size, window_step, prediction_mode):

    instance = Database_instance()
    instance.loading_data_function = load_data_wav
    instance.load_data(path_to_data.split('_left')[0].split('_right')[0].split('_vocals')[0].split('.')[0] +'_vocals.'+path_to_data.split('.')[-1])
    instance.label_filename = labels_filename
    instance.labels, instance.labels_frame_rate = np.array([[0],[0]]), model_output_sample_rate
    instance.align_number_of_labels_and_data()
    instance.generate_timesteps_for_labels()
    instance.cut_data_and_labels_on_windows(window_size, window_step)
    predict_data_with_the_model(model, [instance], prediction_mode=prediction_mode)
    dict_filename_to_predictions = transform_probabilities_to_original_sample_rate(
        database_instances=[instance],
        path_to_video=path_to_video,
        original_sample_rate=model_output_sample_rate,
        need_save=False)
    return dict_filename_to_predictions

def generate_test_predictions_from_list(list_filenames, path_to_data, model, model_output_sample_rate, path_to_video,
                                        window_size, window_step, path_to_output,prediction_mode):
    for filename in list_filenames:
        path_to_audio=path_to_data+filename.split('.')[0]+'_vocals.wav'
        tmp_dict=generate_test_predictions(path_to_audio,filename, model, model_output_sample_rate, path_to_video, window_size, window_step, prediction_mode)
        tmp_dict[filename+'.csv'].to_csv(path_to_output+filename+'.csv', header=False, index=False)


