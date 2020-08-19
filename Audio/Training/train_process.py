import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from sklearn.metrics import accuracy_score, f1_score

from Audio.Training.models import LSTM_model
from Audio.Training.utils import Database, Metric_calculator
import tensorflow as tf

if __name__ == "__main__":
    path_to_data_train='D:\\Databases\\AffWild2\\MFCC_features\\mfcc_23_n_fft_6400_hop_length_3200\\'
    path_to_labels_train='D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\train\\Aligned_labels_reduced\\sample_rate_5\\'
    train_database=Database(path_to_data=path_to_data_train,
                      path_to_labels=path_to_labels_train,
                      data_filetype='csv',
                      data_postfix='_vocals')
    train_database.load_all_data_and_labels()
    train_database.prepare_data_for_training(window_size=4, window_step=2)
    train_data, train_labels=train_database.get_all_concatenated_cutted_data_and_labels()

    # validation
    path_to_data_validation = 'D:\\Databases\\AffWild2\\MFCC_features\\mfcc_23_n_fft_6400_hop_length_3200\\'
    path_to_labels_validation = 'D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels_reduced\\sample_rate_5\\'
    validation_database = Database(path_to_data=path_to_data_validation,
                              path_to_labels=path_to_labels_validation,
                              data_filetype='csv',
                              data_postfix='_vocals')
    validation_database.load_all_data_and_labels()
    validation_database.prepare_data_for_training(window_size=4, window_step=2)
    validation_data, validation_labels = validation_database.get_all_concatenated_cutted_data_and_labels()

    # expand labels to probabilities
    train_labels=tf.keras.utils.to_categorical(train_labels)
    validation_labels=tf.keras.utils.to_categorical(validation_labels)

    # model
    input_shape=train_data.shape[1:]
    batch_size=128
    epochs=20
    num_classes=7

    model=LSTM_model(input_shape, num_classes)
    model.compile(optimizer='Nadam', loss='categorical_crossentropy')
    for epoch in range(epochs):
        model.fit(train_data, train_labels, epochs=1, batch_size=batch_size)
        # predictions for each instance individually
        for instance in validation_database.data_instances:
            predictions=model.predict(instance.cutted_data)
            metric_calculator= Metric_calculator(predictions, instance.cutted_labels_timesteps, ground_truth=instance.labels)
            metric_calculator.average_cutted_predictions_by_timestep(mode='categorical_probabilities')
            instance.predictions=metric_calculator.predictions
        print('epoch:', epoch,'FG_2020 score, all instances:',Metric_calculator(None, None, None).calculate_FG_2020_categorical_score_across_all_instances(validation_database.data_instances))

    predictions=model.predict(validation_data)
    predictions=np.argmax(predictions, axis=-1)
    validation_labels=np.argmax(validation_labels, axis=-1)
    acc=accuracy_score(validation_labels.reshape((-1)), predictions.reshape((-1)))
    f1_score=f1_score(validation_labels.reshape((-1)), predictions.reshape((-1)), average='macro')
    FG_2020_metric=0.67*f1_score+0.33*acc
    print('acc:', acc)
    print('f1_score', f1_score)
    print('FG_2020 score:', FG_2020_metric)

