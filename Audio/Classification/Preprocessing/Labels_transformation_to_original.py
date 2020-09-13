import cv2
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score

from Audio.Classification.Preprocessing.labels_utils_regression import \
    align_number_videoframes_and_labels_all_data, \
    construct_video_filename_from_label, get_video_frame_rate, extend_sample_rate_of_labels
from Audio.Classification.Training.Database import Database
from Audio.Classification.Training.Generator_audio import predict_data_with_model
from Audio.Classification.Training.Metric_calculator import Metric_calculator
from Audio.Classification.Training.models import LSTM_model, CNN_1D_model
from Audio.Classification.Training.utils import load_data_csv, load_labels, plot_confusion_matrix, load_data_wav

from Audio.Classification.Training.models import CNN_1D_model


def generate_predictions(database, model, need_save=True, path_to_save_predictions=''):

    # calculate metric
    predict_data_with_model(model, database.data_instances, prediction_mode=prediction_mode)

    if need_save:
        if not os.path.exists(path_to_save_predictions):
            os.mkdir(path_to_save_predictions)
        for instance in database.data_instances:
            pd.DataFrame(data=instance.predictions_probabilities).to_csv(
                path_to_save_predictions + instance.filename + '.csv', header=False,
                index=False)


def transform_probabilities_to_original_sample_rate(database_instances, path_to_video, original_sample_rate, need_save=True, path_to_output=''):
    dict_filename_to_aligned_predictions={}
    for instance in database_instances:
        # extending
        predictions=instance.predictions_probabilities
        lbs_filename=instance.filename
        predictions=pd.DataFrame(data=predictions)
        video_filename = construct_video_filename_from_label(path_to_video=path_to_video,
                                                             label_filename=lbs_filename)
        video_frame_rate = get_video_frame_rate(path_to_video + video_filename)

        predictions = extend_sample_rate_of_labels(predictions, original_sample_rate, video_frame_rate)
        predictions = predictions.astype('float32')
        # align to video amount of frames
        cap = cv2.VideoCapture(path_to_video+ video_filename)
        video_frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        aligned_predictions = np.zeros(shape=(video_frame_length, predictions.shape[1]), dtype='float32')
        predictions=predictions.values
        if video_frame_length <= predictions.shape[0]:
            aligned_predictions[:] = predictions[:video_frame_length]
        else:
            aligned_predictions[:predictions.shape[0]] = predictions[:]
            value_to_fill = predictions[-1]
            aligned_predictions[predictions.shape[0]:] = value_to_fill
        if need_save:
            if not os.path.exists(path_to_output):
                os.mkdir(path_to_output)
            f = open(path_to_output + lbs_filename.split('_vocal')[0]+'.csv', 'w')
            f.write('Sample rate:%i' % video_frame_rate + '\n')
            f.close()
            aligned_predictions=pd.DataFrame(data=aligned_predictions)
            aligned_predictions.to_csv(path_to_output+lbs_filename.split('_vocal')[0]+'.csv', header=False, index=False)
            # you need to return also
        dict_filename_to_aligned_predictions[lbs_filename+'.csv']=aligned_predictions
    return dict_filename_to_aligned_predictions



if __name__ == "__main__":
    # data params
    path_to_data = 'D:\\Databases\\AffWild2\\Separated_audios\\'
    path_to_labels_train = 'D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\train\\dropped14_interpolated10\\'
    path_to_labels_validation = 'D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels_reduced\\sample_rate_5\\'
    path_to_video='D:\\Databases\\AffWild2\\Videos\\'
    path_to_output = 'results\\'
    path_to_save_predictions='predictions\\'
    path_to_weights='C:\\Users\\Dresvyanskiy\\Downloads\\best_model_weights_1D_CNN.h5'
    original_sample_rate=5
    window_size=4
    window_step=window_size/5.*2.
    num_classes = 7
    prediction_mode='sequence_to_one'
    class_weights_mode = 'scikit'


    # validation data
    validation_database = Database(path_to_data=path_to_data,
                                   path_to_labels=path_to_labels_validation,
                                   data_filetype='wav',
                                   data_postfix='_vocals')
    validation_database.load_all_data_and_labels(loading_data_function=load_data_wav,
                                                 loading_labels_function=load_labels)
    validation_database.prepare_data_for_training(window_size=window_size, window_step=window_step,
                                                  delete_value=None,
                                                  need_scaling=False,
                                                  scaler=None,
                                                  return_scaler=False)

    # model params
    model_input = (validation_database.data_instances[0].data_window_size,) + validation_database.data_instances[0].data.shape[1:]

    optimizer = tf.keras.optimizers.Nadam()
    loss = tf.keras.losses.categorical_crossentropy
    # create model
    model = CNN_1D_model(model_input, num_classes)
    model.load_weights(path_to_weights)
    if prediction_mode == 'sequence_to_sequence':
        model.compile(optimizer=optimizer, loss=loss, sample_weight_mode="temporal")
    else:
        model.compile(optimizer=optimizer, loss=loss)

    generate_predictions(validation_database, model, need_save=False)

    path_to_aligned_extended_predictions=path_to_save_predictions + 'prediction_probabilities_extended_aligned\\'
    path_to_aligned_labels='D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels\\'

    dict_filename_to_predictions=transform_probabilities_to_original_sample_rate(database_instances=validation_database.data_instances,
                                                                                 path_to_video=path_to_video,
                                                                                 original_sample_rate=original_sample_rate,
                                                                                 need_save=True,
                                                                                 path_to_output=path_to_aligned_extended_predictions)

    # calculate score for extended predictions
    real_filenames=os.listdir(path_to_aligned_labels)
    total_predictions=pd.DataFrame()
    total_labels=pd.DataFrame()
    for real_labels_filename in real_filenames:
        predictions_filename=real_labels_filename.split('.')[0].split('_right')[0].split('_left')[0]+'_vocals.csv'
        predictions=dict_filename_to_predictions[predictions_filename]
        if total_predictions.shape[0]==0:
            total_predictions=predictions
        else:
            total_predictions=total_predictions.append(predictions)

        real_labels=pd.read_csv(path_to_aligned_labels+real_labels_filename, header=None)
        if total_labels.shape[0]==0:
            total_labels=real_labels
        else:
            total_labels=total_labels.append(real_labels)

    print(total_predictions.shape)
    total_predictions=np.argmax(total_predictions.values, axis=-1)
    print('final_metric:',0.67*f1_score(total_labels, total_predictions, average='macro')+0.33*accuracy_score(total_labels, total_predictions))
    print('F1:',f1_score(total_labels, total_predictions, average='macro'))
    print('accuracy:',accuracy_score(total_labels, total_predictions))