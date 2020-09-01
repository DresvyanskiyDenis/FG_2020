import os
import pandas as pd
import numpy as np
from sklearn.utils import class_weight

from Audio.Training.Metric_calculator import Metric_calculator
from Audio.Training.models import CNN_1D_model
from Audio.Training.Database import Database
from Audio.Training.Generator_audio import batch_generator_cut_data, predict_data_with_model
from Audio.Training.utils import load_labels, load_data_wav, generate_weights
import tensorflow as tf


def train_model_on_data(path_to_data, path_to_labels_train, path_to_labels_validation, path_to_output, window_size, window_step, class_weights_mode='my_realisation'):
    # data params
    path_to_data_train = path_to_data
    path_to_labels_train = path_to_labels_train

    train_database = Database(path_to_data=path_to_data_train,
                              path_to_labels=path_to_labels_train,
                              data_filetype='wav',
                              data_postfix='_vocals')
    train_database.load_all_data_and_labels(loading_data_function=load_data_wav, loading_labels_function=load_labels)
    train_database.prepare_data_for_training(window_size=window_size, window_step=window_step,
                                                               need_scaling=False,
                                                               scaler=None,
                                                               return_scaler=False)

    # validation data
    validation_database = Database(path_to_data=path_to_data_train,
                              path_to_labels=path_to_labels_validation,
                              data_filetype='wav',
                              data_postfix='_vocals')
    validation_database.load_all_data_and_labels(loading_data_function=load_data_wav, loading_labels_function=load_labels)
    validation_database.prepare_data_for_training(window_size=window_size, window_step=window_step,
                                                                 delete_value=None,
                                                                 need_scaling=False,
                                                                 scaler=None,
                                                                 return_scaler=False)




    # model params
    model_input=(train_database.data_instances[0].data_window_size,)+train_database.data_instances[0].data.shape[1:]
    num_classes=7
    batch_size=20
    epochs=100
    optimizer=tf.keras.optimizers.Nadam()
    loss=tf.keras.losses.categorical_crossentropy
    # create model
    model= CNN_1D_model(model_input, num_classes)
    model.compile(optimizer=optimizer, loss=loss, sample_weight_mode="temporal")

    # class weighting through sample weighting, while keras do not allow use class_weights with reccurent layers and 3D+ data
    if class_weights_mode == 'my_realisation':
        class_weights = generate_weights(
            np.unique(train_database.get_all_concatenated_data_and_labels()[1], return_counts=True)[1])
    elif class_weights_mode == 'scikit':
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(
                                                              train_database.get_all_concatenated_data_and_labels()[1]),
                                                          train_database.get_all_concatenated_data_and_labels()[1])

    best_result=0
    for epoch in range(epochs):
        train_generator = batch_generator_cut_data(train_database.data_instances, need_shuffle=True,
                                                   batch_size=batch_size, need_sample_weight=True, class_weights=class_weights)
        num_batch=0
        loss_sum=0
        for generator_step in train_generator:
            train_data, train_labels, sample_weights=generator_step
            train_labels=tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
            train_data, train_labels, sample_weights=train_data.astype('float32'), train_labels.astype('float32'), sample_weights.astype('float32')
            train_result=model.train_on_batch(train_data, train_labels, sample_weight=sample_weights)
            #print('Epoch %i, num batch:%i, loss:%f'%(epoch, num_batch,train_result))
            num_batch+=1
            loss_sum+=train_result
        # calculate metric on validation
        predict_data_with_model(model, validation_database.data_instances)
        validation_result = Metric_calculator(None, None,None).\
            calculate_FG_2020_categorical_score_across_all_instances(validation_database.data_instances)
        print('Epoch %i is ended. Average loss:%f, validation FG-2020 metric:%f' % (epoch, loss_sum / num_batch, validation_result))
        if validation_result>=best_result:
            best_result=validation_result
            model.save_weights(path_to_output+'best_model_weights.h5')
    results=pd.DataFrame(columns=['data directory', 'window size', 'validation_result'])
    results=results.append({'data directory':path_to_data, 'window size':window_size, 'validation_result':best_result}, ignore_index=True)
    results.to_csv(path_to_output+'test_results.csv', index=False)
    return best_result



if __name__ == "__main__":
    # data params
    path_to_data='D:\\Databases\\AffWild2\\Separated_audios\\'
    path_to_labels_train='D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\train\\dropped14_interpolated10\\'
    path_to_labels_validation = 'D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels_reduced\\sample_rate_5\\'
    path_to_output='results\\'
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
    data_directories=os.listdir(path_to_data)
    window_sizes=[4]
    results=pd.DataFrame(columns=['data directory', 'window size', 'validation_result'])
    for window_size in window_sizes:
        output_directory=path_to_output+'1D_CNN_window_size_'+str(window_size)+'\\'
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        val_result=train_model_on_data(path_to_data=path_to_data,
                                       path_to_labels_train=path_to_labels_train,
                                       path_to_labels_validation=path_to_labels_validation,
                                       path_to_output=output_directory,
                                       window_size=window_size,
                                       window_step=window_size*2./5.,
                                       class_weights_mode='my_realisation')
        results=results.append({'data directory':path_to_data, 'window size':window_size, 'validation_result':val_result}, ignore_index=True)
        results.to_csv('test_results.csv', index=False)

