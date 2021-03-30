import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from scipy.special import softmax

from Fusion.Classification.prediction_fusion import align_sample_rate_to_video_rate


def main():
    path_to_model_3_predictions = 'C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\Pretrained_AffectNet_model_3\\test'
    path_to_model_4_predictions = 'C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\Pretrained_AffectNet_model_4\\test'
    path_to_1D_CNN_predictions = 'C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\1D_CNN\\test'
    path_to_PANN_predictions = 'C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\PANN\\test'
    path_to_linearSVM_predictions = 'C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\linearSVM_4_2_z_l2_C_0_1\\test\\trained_only_train'
    path_to_videos='E:\\Databases\\AffWild2\\Videos\\'
    path_to_fusion_weights='C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\predictions_for_submission\\' \
                           'Elena_AffectNet_3_4_and_1DCNN_LinearSVM_4_2_z_l2_C_0_1\\weights_for_fusion.txt'
    path_to_output_test_predictions='predictions'
    if not os.path.exists(path_to_output_test_predictions):
        os.makedirs(path_to_output_test_predictions, exist_ok=True)

    # load all filenames with predictions
    labels_filenames = os.listdir(path_to_model_3_predictions)
    for lbs_filename in labels_filenames:
        filename = lbs_filename.split('.')[0]

        model_3_pred = pd.read_csv(os.path.join(path_to_model_3_predictions, filename + '.txt'))
        model_3_pred = pd.DataFrame(data=model_3_pred.iloc[:, 1:].values)

        model_4_pred = pd.read_csv(os.path.join(path_to_model_4_predictions, filename + '.txt'))
        model_4_pred = pd.DataFrame(data=model_4_pred.iloc[:, 1:].values)

        model_PANN_pred = pd.read_csv(os.path.join(path_to_PANN_predictions, filename + '.csv'), header=None)
        model_PANN_pred = align_sample_rate_to_video_rate(model_PANN_pred, path_to_videos, filename, 5)

        model_1D_pred = pd.read_csv(os.path.join(path_to_1D_CNN_predictions, filename + '.csv'), header=None)
        model_1D_pred = align_sample_rate_to_video_rate(model_1D_pred, path_to_videos, filename, 5)

        model_linearSVM_pred = pd.read_csv(os.path.join(path_to_linearSVM_predictions, filename + '.csv'))
        model_linearSVM_pred.iloc[:]=softmax(model_linearSVM_pred.iloc[:], axis=1)
        model_linearSVM_pred = pd.DataFrame(data=model_linearSVM_pred.values)

        # define, which predictions should be included in this try
        predictions = [model_3_pred, model_4_pred, model_1D_pred, model_linearSVM_pred]
        num_predictions = len(predictions)
        # load weights
        weights = np.loadtxt(path_to_fusion_weights)
        # calculare final predictions by weighted fusion
        final_prediction = predictions[0] * weights[:, 0]
        for i in range(1, num_predictions):
            final_prediction += predictions[i] * weights[:, i]
        # make final fusion as argmax of probabilities
        final_prediction = np.argmax(final_prediction.values, axis=-1)
        final_prediction = final_prediction[..., np.newaxis]
        # save generated predictions
        columns_for_test='Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise\n'
        file = open(os.path.join(path_to_output_test_predictions, filename + '.txt'), 'w')
        file.write(columns_for_test)
        file.close()
        pd.DataFrame(data=final_prediction).to_csv(os.path.join(path_to_output_test_predictions, filename + '.txt'),
                                                   header=False, index=False, mode='a')




if __name__=='__main__':
    main()