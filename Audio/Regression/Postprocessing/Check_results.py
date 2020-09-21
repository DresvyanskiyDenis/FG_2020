import tensorflow as tf

from Audio.utils.Database import Database
from Audio.utils.Generator_audio import predict_data_with_the_model
from Audio.utils.Metric_calculator import Metric_calculator
from Audio.utils.models import sequence_to_sequence_regression_model
from Audio.utils.utils import load_data_wav, load_labels, CCC_loss_tf

if __name__ == "__main__":
    # data params
    path_to_data='D:\\Databases\\AffWild2\\Separated_audios\\'
    path_to_labels_train='D:\\Databases\\AffWild2\\Annotations\\VA_Set\\train\\Aligned_labels_reduced_without_outliers\\'
    path_to_labels_validation = 'D:\\Databases\\AffWild2\\Annotations\\VA_Set\\validation\\Aligned_labels_reduced_separated\\arousal\\'
    path_to_output= 'results\\'
    path_to_weights='C:\\Users\\Dresvyanskiy\\Downloads\\Arousal_seq_to_seq_weights.h5'
    window_size=4
    window_step=window_size/5.*2.
    prediction_mode='sequence_to_sequence'
    class_weights_mode = 'scikit'

    '''train_database = Database(path_to_data=path_to_data,
                              path_to_labels=path_to_labels_train,
                              data_filetype='wav',
                              data_postfix='_vocals')
    train_database.load_all_data_and_labels(loading_data_function=load_data_wav, loading_labels_function=load_labels)
    train_database.prepare_data_for_training(window_size=window_size, window_step=window_step,
                                             need_scaling=False,
                                             scaler=None,
                                             return_scaler=False)'''

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
    batch_size = 20
    epochs = 2
    optimizer = tf.keras.optimizers.Nadam()
    loss = tf.keras.losses.categorical_crossentropy
    # create model
    model = sequence_to_sequence_regression_model(model_input)
    model.load_weights(path_to_weights)

    if prediction_mode == 'sequence_to_sequence':
        model.compile(optimizer=optimizer, loss=CCC_loss_tf)
    else:
        model.compile(optimizer=optimizer, loss=CCC_loss_tf)

    predict_data_with_the_model(model, validation_database.data_instances, prediction_mode=prediction_mode,
                                labels_type='regression')

    validation_result = Metric_calculator.calculate_FG_2020_CCC_score_with_extended_predictions(
        instances=validation_database.data_instances,
        path_to_video='D:\\Databases\\AffWild2\\Videos\\',
        path_to_real_labels='D:\\Databases\\AffWild2\\Annotations\\VA_Set\\validation\\Aligned_labels_separated\\arousal\\',
        original_sample_rate=10,
        delete_value=-5)
    print('Arousal CCC:', validation_result)



