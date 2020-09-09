import os
import pandas as pd
import numpy as np
from sklearn.utils import class_weight

from Audio.Classification.Training.Metric_calculator import Metric_calculator
from Audio.Classification.Training.models import CNN_1D_model
from Audio.Classification.Training.Database import Database
from Audio.Classification.Training.Generator_audio import batch_generator_cut_data, predict_data_with_model
from Audio.Classification.Training.utils import load_labels, load_data_wav, generate_weights, find_the_greatest_class_in_array
import tensorflow as tf


from Audio.Classification.Training.Database import Database
from Audio.Classification.Training.Generator_audio import predict_data_with_model
from Audio.Classification.Training.Metric_calculator import Metric_calculator
from Audio.Classification.Training.models import LSTM_model, CNN_1D_model
from Audio.Classification.Training.utils import load_data_csv, load_labels, plot_confusion_matrix, load_data_wav

if __name__ == "__main__":
    # data params
    # data params
    path_to_data = 'D:\\Databases\\AffWild2\\Separated_audios\\'
    path_to_labels_train = 'D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\train\\dropped14_interpolated10\\'
    path_to_labels_validation = 'D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels_reduced\\sample_rate_5\\'
    path_to_output = 'results\\'
    path_to_weights='C:\\Users\\Dresvyanskiy\\Downloads\\best_model_weights_1D_CNN.h5'
    window_size=4
    window_step=window_size/5.*2.
    prediction_mode='sequence_to_one'
    class_weights_mode = 'scikit'

    train_database = Database(path_to_data=path_to_data,
                              path_to_labels=path_to_labels_train,
                              data_filetype='wav',
                              data_postfix='_vocals')
    train_database.load_all_data_and_labels(loading_data_function=load_data_wav, loading_labels_function=load_labels)
    train_database.prepare_data_for_training(window_size=window_size, window_step=window_step,
                                             need_scaling=False,
                                             scaler=None,
                                             return_scaler=False)

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
    model_input = (train_database.data_instances[0].data_window_size,) + train_database.data_instances[0].data.shape[1:]
    num_classes = 7
    batch_size = 20
    epochs = 2
    optimizer = tf.keras.optimizers.Nadam()
    loss = tf.keras.losses.categorical_crossentropy
    # create model
    model = CNN_1D_model(model_input, num_classes)
    model.load_weights(path_to_weights)
    if prediction_mode == 'sequence_to_sequence':
        model.compile(optimizer=optimizer, loss=loss, sample_weight_mode="temporal")
    else:
        model.compile(optimizer=optimizer, loss=loss)

    # class weighting through sample weighting, while keras do not allow use class_weights with reccurent layers and 3D+ data
    if class_weights_mode == 'my_realisation':
        class_weights = generate_weights(
            np.unique(train_database.get_all_concatenated_data_and_labels()[1], return_counts=True)[1])
    elif class_weights_mode == 'scikit':
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(
                                                              train_database.get_all_concatenated_data_and_labels()[1]),
                                                          train_database.get_all_concatenated_data_and_labels()[1])

    # calculate metric on train
    predict_data_with_model(model, train_database.data_instances, prediction_mode=prediction_mode)
    train_result = Metric_calculator(None, None, None). \
        calculate_FG_2020_F1_and_accuracy_scores_across_all_instances(train_database.data_instances)
    print('Train: FG_2020 metric:%f, F1:%f, Accuracy:%f'%train_result)
    train_database.plot_confusion_matrix()

    # calculate metric on validation
    predict_data_with_model(model, validation_database.data_instances, prediction_mode=prediction_mode)
    validation_result = Metric_calculator(None, None, None). \
        calculate_FG_2020_F1_and_accuracy_scores_across_all_instances(validation_database.data_instances)
    print('Validation: FG_2020 metric:%f, F1:%f, Accuracy:%f' % validation_result)
    validation_database.plot_confusion_matrix()

