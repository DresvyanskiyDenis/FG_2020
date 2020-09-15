import os
import numpy as np
import tensorflow as tf

from Audio.Classification.Training.models import CNN_1D_model
from Audio.Classification.Training.utils import generate_test_predictions_from_list

if __name__ == "__main__":
    path_to_filenames_labels='C:\\Users\\Dresvyanskiy\\Desktop\\expression_test_set.txt'
    #filenames=pd.read_csv(path_to_filenames_labels, header=None)
    filenames=np.array(os.listdir('D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels_reduced\\sample_rate_5\\'))
    filenames=filenames.reshape((-1,))
    path_to_weights='C:\\Users\\Dresvyanskiy\\Downloads\\best_model_weights_1D_CNN.h5'
    path_to_data='D:\\Databases\\AffWild2\\Separated_audios\\'
    path_to_output= 'predictions_val\\'
    path_to_video='D:\\Databases\\AffWild2\\Videos\\'
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
    window_size=4
    window_step=window_size/5.*2.
    num_classes = 7
    prediction_mode='sequence_to_one'
    model_output_sample_rate=5
    # model params
    model_input = (window_size*16000,1)
    optimizer = tf.keras.optimizers.Nadam()
    loss = tf.keras.losses.categorical_crossentropy
    # create model
    model = CNN_1D_model(model_input, num_classes)
    model.load_weights(path_to_weights)
    if prediction_mode == 'sequence_to_sequence':
        model.compile(optimizer=optimizer, loss=loss, sample_weight_mode="temporal")
    else:
        model.compile(optimizer=optimizer, loss=loss)

    generate_test_predictions_from_list(list_filenames=filenames,
                                        path_to_data=path_to_data,
                                        model=model, model_output_sample_rate=model_output_sample_rate,
                                        path_to_video=path_to_video,
                                        window_size=window_size, window_step=window_step,
                                        path_to_output=path_to_output,
                                        prediction_mode=prediction_mode)