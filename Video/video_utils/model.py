import tensorflow as tf

from Audio.utils.utils import CCC_loss_tf
from Video.video_utils.VGG_face2.src.model import Vggface2_ResNet50


def create_and_load_pretrained_VGGFace2_model(input_shape, path_to_weights):
    model=Vggface2_ResNet50(input_dim=input_shape, mode='train')
    #model.load_weights(path_to_weights, by_name=True)
    new_model=tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer('dim_proj').output])
    new_model.load_weights(path_to_weights, by_name=True)
    del model
    return new_model


def create_seq_to_seq_regression_model(input_shape, pretrained_model=None):
    if pretrained_model==None:
        raise Exception('You did not pass any pretrained model')
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.TimeDistributed(pretrained_model, input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh')))
    optimizer=tf.keras.optimizers.Adam(lr=0.00025)
    loss=CCC_loss_tf
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()
    return model