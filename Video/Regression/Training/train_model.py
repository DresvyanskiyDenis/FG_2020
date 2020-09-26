import os
import pandas as pd
import numpy as np
from sklearn.utils import class_weight

import tensorflow as tf

from Video.video_utils.batcher import frames_batch_generator, calculate_model_performance_by_path_to_validation_batches
from Video.video_utils.model import create_and_load_pretrained_VGGFace2_model, create_seq_to_seq_regression_model


def main():
    path_to_batches='D:\\Databases\\AffWild2\\batches_train\\'
    path_to_weights_model='C:\\Users\\Dresvyanskiy\\Downloads\\vggface2_Keras (1)\\vggface2_Keras\\model\\resnet50_softmax_dim512\\weights.h5'
    path_to_batches_validation='D:\\Databases\\AffWild2\\1\\'
    path_to_save_model='results\\'
    input_shape=(30,224,224,3)
    epochs=100
    label_type=['valence', 'arousal']
    pretrained_model=create_and_load_pretrained_VGGFace2_model(input_shape=input_shape[1:], path_to_weights=path_to_weights_model)
    model=create_seq_to_seq_regression_model(input_shape=input_shape, pretrained_model=pretrained_model)
    batcher=frames_batch_generator(path_to_batches=path_to_batches,
                                   batch_size=1,
                                   data_prefix='data_batch_num_',
                                   labels_prefix='labels_timesteps_batch_num_',
                                   labels_type=label_type)

    if not os.path.exists(path_to_save_model):
        os.mkdir(path_to_save_model)
    best=0
    for epoch in range(epochs):
        loss=0
        counter_batches=0
        for batch in batcher:
            data, labels, timesteps = batch
            loss+=model.train_on_batch(data, labels)
            counter_batches+=1
            if counter_batches % 100 == 0: print(
                'mini batch number %i, average_loss:%f' % (counter_batches, loss / counter_batches))
        loss=loss/counter_batches
        results = calculate_model_performance_by_path_to_validation_batches(model=model,
                                                                            path_to_batches=path_to_batches_validation,
                                                                            label_type=label_type)
        print('epoch:%i, average loss:%f, validation performance:%f'%(epoch, loss, np.array(results).mean()))
        if np.array(results).mean()>=best:
            model.save_weights(path_to_save_model+'model_best_weights.h5')
            best=np.array(results).mean()
            pd.DataFrame(columns=['best_value'], data=np.array(best)).to_csv(path_to_save_model + 'best_result.csv',
                                                                             index=False)



if __name__ == "__main__":
    main()