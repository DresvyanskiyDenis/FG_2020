import os
import pandas as pd
import numpy as np
from sklearn.utils import class_weight

import tensorflow as tf

from Video.video_utils.batcher import frames_batch_generator, calculate_model_performance_by_path_to_validation_batches
from Video.video_utils.model import create_and_load_pretrained_VGGFace2_model, create_seq_to_seq_regression_model


def main():
    path_to_batches='D:\\Databases\\AffWild2\\batches_train\\'
    path_to_weights_model='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\Video\\video_utils\\VGG_face2\\model\\resnet50_softmax_dim512\\weights.h5'
    path_to_batches_validation='D:\\Databases\\AffWild2\\1\\'
    input_shape=(30,224,224,3)
    epochs=100
    pretrained_model=create_and_load_pretrained_VGGFace2_model(input_shape=input_shape[1:], path_to_weights=path_to_weights_model)
    model=create_seq_to_seq_regression_model(input_shape=input_shape, pretrained_model=pretrained_model)
    batcher=frames_batch_generator(path_to_batches=path_to_batches,
                                   batch_size=1,
                                   image_shape=(224,224,3),
                                   data_prefix='data_batch_num_',
                                   labels_prefix='labels_timesteps_batch_num_')
    results=calculate_model_performance_by_path_to_validation_batches(model=model,
                                                                      path_to_batches=path_to_batches_validation,
                                                                      label_type=['valence', 'arousal'])
    print(results)
    for epoch in range(epochs):
        loss=0
        counter_batches=0
        for batch in batcher:
            loss+=model.train_on_batch(batch)
            counter_batches+=1
        loss=loss/counter_batches
        print('epoch:%i, average loss:%f'%(epoch, loss))











if __name__ == "__main__":
    main()