import gc
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight
from tensorflow.keras import backend as K

from Audio.Training.Database import Database
from Audio.Training.Metric_calculator import Metric_calculator
from Audio.Training.models import LSTM_model
from Audio.Training.utils import generate_weights, load_data_csv, load_labels, plot_confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def train_model_on_data(path_to_data, path_to_labels_train, path_to_labels_validation, path_to_output, window_size, window_step, class_weights_mode='my_realisation'):
    # data params
    path_to_data_train = path_to_data
    path_to_labels_train = path_to_labels_train

    train_database = Database(path_to_data=path_to_data_train,
                              path_to_labels=path_to_labels_train,
                              data_filetype='csv',
                              data_postfix='_vocals')
    train_database.load_all_data_and_labels(loading_data_function=load_data_csv, loading_labels_function=load_labels)
    train_data_scaler=train_database.prepare_data_for_training(window_size=window_size, window_step=window_step,
                                                               need_scaling=True,
                                                               scaler=None,
                                                               return_scaler=True)
    train_data, train_labels = train_database.get_all_concatenated_cutted_data_and_labels()

    # validation
    path_to_data_validation = path_to_data
    path_to_labels_validation = path_to_labels_validation
    validation_database = Database(path_to_data=path_to_data_validation,
                                   path_to_labels=path_to_labels_validation,
                                   data_filetype='csv',
                                   data_postfix='_vocals')
    validation_database.load_all_data_and_labels(loading_data_function=load_data_csv, loading_labels_function=load_labels)
    validation_database.prepare_data_for_training(window_size=window_size, window_step=window_step,
                                                               delete_value=None,
                                                               need_scaling=True,
                                                               scaler=train_data_scaler,
                                                               return_scaler=False)
    validation_data, validation_labels = validation_database.get_all_concatenated_cutted_data_and_labels()

    # expand labels to probabilities
    train_labels = tf.keras.utils.to_categorical(train_labels)
    validation_labels = tf.keras.utils.to_categorical(validation_labels)

    # model
    input_shape = train_data.shape[1:]
    batch_size = 256
    epochs = 20
    num_classes = 7

    model = LSTM_model(input_shape, num_classes)
    model.compile(optimizer='Nadam', loss='categorical_crossentropy', sample_weight_mode="temporal")

    # class weighting through sample weighting, while keras do not allow use class_weights with reccurent layers and 3D+ data
    if class_weights_mode=='my_realisation':
        class_weights = generate_weights(
            np.unique(train_database.get_all_concatenated_data_and_labels()[1], return_counts=True)[1])
    elif class_weights_mode =='scikit':
        class_weights=class_weight.compute_class_weight('balanced',
                                    np.unique(train_database.get_all_concatenated_data_and_labels()[1]),
                                    train_database.get_all_concatenated_data_and_labels()[1])

    sample_weight = np.argmax(train_labels, axis=-1).astype('float32')
    for i in range(class_weights.shape[0]):
        mask = (sample_weight == i)
        sample_weight[mask] = class_weights[i]

    best_model = None
    best_result = 100000000
    for epoch in range(epochs):
        # shuffle data
        permutations=np.random.permutation(train_data.shape[0])
        train_data, train_labels, sample_weight = train_data[permutations], train_labels[permutations], sample_weight[permutations]

        model.fit(train_data, train_labels, epochs=1, batch_size=batch_size, sample_weight=sample_weight)
        # predictions for each instance individually
        for instance in validation_database.data_instances:
            predictions = model.predict(instance.cutted_data)
            metric_calculator = Metric_calculator(predictions, instance.cutted_labels_timesteps,
                                                  ground_truth=instance.labels)
            metric_calculator.average_cutted_predictions_by_timestep(mode='categorical_probabilities')
            instance.predictions = metric_calculator.predictions
        validation_result = Metric_calculator(None, None,
                                              None).calculate_FG_2020_categorical_score_across_all_instances(
            validation_database.data_instances)
        print('epoch:', epoch, 'FG_2020 score, all instances:', validation_result)
        if validation_result <= best_result:
            best_result = validation_result
            model.save_weights(path_to_output+'best_model_weights.h5')

    # check the performance of best model and plot confusion matrix
    model = LSTM_model(input_shape, num_classes)
    model.load_weights(path_to_output+'best_model_weights.h5')
    model.compile(optimizer='Nadam', loss='categorical_crossentropy', sample_weight_mode="temporal")
    for instance in validation_database.data_instances:
        predictions = model.predict(instance.cutted_data)
        metric_calculator = Metric_calculator(predictions, instance.cutted_labels_timesteps,
                                              ground_truth=instance.labels)
        metric_calculator.average_cutted_predictions_by_timestep(mode='categorical_probabilities')
        instance.predictions = metric_calculator.predictions

    validation_result = Metric_calculator(None, None, None).calculate_FG_2020_categorical_score_across_all_instances(
        validation_database.data_instances)
    print('best_model FG_2020 score, all instances:', validation_result)
    ground_truth_all = np.zeros((0,))
    predictions_all = np.zeros((0,))
    for instance in validation_database.data_instances:
        ground_truth_all = np.concatenate((ground_truth_all, instance.labels))
        predictions_all = np.concatenate((predictions_all, instance.predictions))

    ax=plot_confusion_matrix(y_true=ground_truth_all,
                          y_pred=predictions_all,
                          classes=np.unique(ground_truth_all),
                          path_to_save=path_to_output)


    # clear RAM
    del model
    K.clear_session()
    gc.collect()
    return validation_result


if __name__ == "__main__":
    # data params
    path_to_data='D:\\Databases\\AffWild2\\MFCC_features\\'
    path_to_labels_train='D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\train\\dropped14_interpolated10\\'
    path_to_labels_validation = 'D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels_reduced\\sample_rate_5\\'
    path_to_output='results\\'
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
    data_directories=os.listdir(path_to_data)
    window_sizes=[2,4,6, 12]
    results=pd.DataFrame(columns=['data directory', 'window size', 'validation_result'])
    for directory in data_directories:
        for window_size in window_sizes:
            output_directory=path_to_output+directory+'_window_size_'+str(window_size)+'\\'
            if not os.path.exists(output_directory):
                os.mkdir(output_directory)
            val_result=train_model_on_data(path_to_data=path_to_data+directory+'\\',
                                           path_to_labels_train=path_to_labels_train,
                                           path_to_labels_validation=path_to_labels_validation,
                                           path_to_output=output_directory,
                                           window_size=window_size,
                                           window_step=window_size*2./5.,
                                           class_weights_mode='my_realisation')
            results=results.append({'data directory':directory, 'window size':window_size, 'validation_result':val_result}, ignore_index=True)
            results.to_csv('test_results.csv', index=False)

