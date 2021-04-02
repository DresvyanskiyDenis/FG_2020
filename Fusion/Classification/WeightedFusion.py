import os
from itertools import product
from typing import List

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from scipy.special import softmax

from Fusion.Classification.prediction_fusion import align_sample_rate_to_video_rate

def plot_and_save_confusion_matrix(y_true, y_pred, name_labels, path_to_save:str='confusion_matrix', name_filename:str='cm.png', title:str=''):
    c_m = confusion_matrix(y_true, y_pred)
    conf_matrix = pd.DataFrame(c_m, name_labels, name_labels)

    plt.figure(figsize=(10, 10))
    #plt.title(title, y=1., fontsize=20)

    group_counts = ['{0:0.0f}'.format(value) for value in
                    conf_matrix.values.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         conf_matrix.div(np.sum(conf_matrix, axis=1), axis=0).values.flatten()]

    labels = ['{}\n{}'.format(v1, v2) for v1, v2 in zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(c_m.shape)
    sns.set(font_scale=1.6)
    chart = sns.heatmap(conf_matrix,
                        cbar=False,
                        annot=labels,
                        square=True,
                        fmt='',
                        annot_kws={'size': 18},
                        cmap="Blues"
                        )
    chart.set_title(title, fontsize=20)
    chart.set_yticklabels(labels=chart.get_yticklabels(), va='center')
    chart.set_xticklabels(labels=chart.get_xticklabels(), va='center')
    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    plt.savefig(os.path.join(path_to_save,name_filename), bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

def plot_weights(weights, path_to_save:str='confusion_matrix', name_filename:str='cm.png', title:str=''):
    weights=weights.T
    weights=pd.DataFrame(data=weights, index=['AffectNet','AffWild2','Audio M.', 'L-SVM'],
                         columns=['Ne','An','Di','Fe', 'Ha','Sad','Sur'])
    #plt.figure(figsize=(10, 10))
    group_percentages = ['{0:.2}'.format(value) for value in
                         weights.values.flatten()]
    labels = ['{}'.format(v1) for v1 in group_percentages]
    labels = np.asarray(labels).reshape(weights.shape)
    sns.set(font_scale=1.3)
    chart = sns.heatmap(weights,
                        cbar=False,
                        annot=labels,
                        square=True,
                        fmt='',
                        annot_kws={'size': 15},
                        cmap="Blues"
                        )
    chart.set_yticklabels(labels=chart.get_yticklabels(), va='center')
    chart.set_xticklabels(labels=chart.get_xticklabels(), va='center')
    #chart.set_title(title, fontsize=14)
    #plt.ylabel("Class weights")
    #plt.xlabel("Models")
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    plt.savefig(os.path.join(path_to_save, name_filename), bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

def plot_f1_scores_of_models_from_predictions(predictions:List[np.ndarray], ground_truth:np.array, path_to_save:str='confusion_matrix',
                                                   name_filename:str='cm.png', title:str='', num_classes:int=7):
    ground_truth=ground_truth.reshape((-1,1))

    f1_scores=[]
    for pred_idx in range(len(predictions)):
        curr_predictions=predictions[pred_idx].reshape((-1,1))
        f1=[]
        for class_idx in range(num_classes):
            f1.append(f1_score(ground_truth, curr_predictions, average='macro', labels=[class_idx]))
        f1_scores.append(f1)



    f1_scores=np.array(f1_scores)
    labels=np.array(['{0:.2}'.format(value) for value in f1_scores.reshape((-1,))]).reshape(f1_scores.shape)
    f1_scores=pd.DataFrame(f1_scores, columns=['Ne','An','Di','Fe',
                                                    'Ha','Sad','Sur'],
                         index=['AffectNet','AffWild2','Audio M.', 'L-SVM', 'Fusion'])

    sns.set(font_scale=1.3)
    chart = sns.heatmap(f1_scores,
                        cbar=False,
                        annot=labels,
                        square=True,
                        fmt='',
                        annot_kws={'size': 15},
                        cmap="Blues"
                        )
    chart.set_yticklabels(labels=chart.get_yticklabels(), va='center')
    chart.set_xticklabels(labels=chart.get_xticklabels(), va='center')
    #chart.set_title(title, fontsize=16)
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    plt.savefig(os.path.join(path_to_save, name_filename), bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()








def main():
    path_to_model_3_predictions = 'C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\Predictions\\Pretrained_AffectNet_model_3\\devel'
    path_to_model_4_predictions = 'C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\Predictions\\Pretrained_AffectNet_model_4\\devel'
    path_to_1D_CNN_predictions = 'C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\Predictions\\1D_CNN\\devel'
    path_to_PANN_predictions = 'C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\Predictions\\PANN\\devel'
    path_to_linearSVM_predictions = 'C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\Predictions\\linearSVM_4_2_z_power_l2_C_0_01\\validation'
    path_to_ground_truth_predictions='D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels\\'
    path_to_videos='D:\\Databases\\AffWild2\\Videos\\'
    path_to_fusion_weights='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\Predictions\\predictions_for_submission\\fused_svm_train_dev_power_submission_11\\weights_for_fusion.txt'
    num_classes = 7

    # load all filenames with predictions
    labels_filenames = os.listdir(path_to_model_3_predictions)
    # create variables to load predictions
    model_3_predictions = []
    model_4_predictions = []
    model_1D_predictions = []
    model_PANN_predictions =[]
    model_linearSVM_predictions=[]
    ground_truth_predictions = []
    for lbs_filename in labels_filenames:
        filename = lbs_filename.split('.')[0]
        ground_truth_pred = pd.read_csv(os.path.join(path_to_ground_truth_predictions, filename + '.txt'), header=None)

        model_3_pred = pd.read_csv(os.path.join(path_to_model_3_predictions, filename + '.txt'))
        model_3_pred = pd.DataFrame(data=model_3_pred.iloc[:, 1:].values)

        model_4_pred = pd.read_csv(os.path.join(path_to_model_4_predictions, filename + '.txt'))
        model_4_pred = pd.DataFrame(data=model_4_pred.iloc[:, 1:].values)

        model_PANN_pred = pd.read_csv(os.path.join(path_to_PANN_predictions, filename + '.csv'), header=None)
        model_PANN_pred = align_sample_rate_to_video_rate(model_PANN_pred, path_to_videos, filename, 5)

        model_1D_pred = pd.read_csv(os.path.join(path_to_1D_CNN_predictions, filename + '.csv'), header=None)
        model_1D_pred = align_sample_rate_to_video_rate(model_1D_pred, path_to_videos, filename, 5)

        model_linearSVM_pred = pd.read_csv(os.path.join(path_to_linearSVM_predictions, filename + '.csv'))
        #model_linearSVM_pred = align_sample_rate_to_video_rate(model_linearSVM_pred, path_to_videos, filename, 5)
        model_linearSVM_pred.iloc[:]=softmax(model_linearSVM_pred.iloc[:], axis=1)
        model_linearSVM_pred = pd.DataFrame(data=model_linearSVM_pred.values)

        # append to the lists created above for former concatenation
        model_3_predictions.append(model_3_pred)
        model_4_predictions.append(model_4_pred)
        model_PANN_predictions.append(model_PANN_pred)
        model_1D_predictions.append(model_1D_pred)
        model_linearSVM_predictions.append(model_linearSVM_pred)
        ground_truth_predictions.append(ground_truth_pred)
    # concatenate all loaded predictions
    model_3_predictions=pd.concat(model_3_predictions)
    model_4_predictions = pd.concat(model_4_predictions)
    model_PANN_predictions = pd.concat(model_PANN_predictions)
    model_1D_predictions = pd.concat(model_1D_predictions)
    model_linearSVM_predictions = pd.concat(model_linearSVM_predictions)
    ground_truth_predictions = pd.concat(ground_truth_predictions)

    # define, which predictions should be included in this try
    predictions = [model_3_predictions, model_4_predictions, model_1D_predictions, model_linearSVM_predictions]
    num_predictions = len(predictions)
    # load weights
    weights = np.loadtxt(path_to_fusion_weights)
    plot_weights(weights,path_to_save='confusion_matrix', name_filename='weights.pdf', title='Weights for fusion')
    # calculare final predictions by weighted fusion
    final_prediction = predictions[0] * weights[:,0]
    for i in range(1, num_predictions):
        final_prediction += predictions[i] * weights[:,i]
    #final_prediction=predictions[3]
    # make final fusion as argmax of probabilities
    final_prediction = np.argmax(final_prediction.values, axis=-1)
    final_prediction = final_prediction[..., np.newaxis]
    # delete labels with -1 class, they should not be taken into account
    delete_mask = ground_truth_predictions.values != -1
    final_prediction=final_prediction[delete_mask]
    plot_f1_scores_of_models_from_predictions(predictions=[
         np.argmax(model_3_predictions.values, axis=-1)[...,np.newaxis][delete_mask],
         np.argmax(model_4_predictions.values, axis=-1)[...,np.newaxis][delete_mask],
         np.argmax(model_1D_predictions.values, axis=-1)[..., np.newaxis][delete_mask],
         np.argmax(model_linearSVM_predictions.values, axis=-1)[..., np.newaxis][delete_mask],
        final_prediction
        ],
        ground_truth=ground_truth_predictions.values[delete_mask],
        path_to_save='confusion_matrix',
        name_filename='f1_scores.pdf', title=None)

    # draw the confusion matrix of predictions
    plot_and_save_confusion_matrix(y_true=ground_truth_predictions.values[delete_mask], y_pred=final_prediction,
                                   name_labels=['Neutral','Anger','Disgust','Fear',
                                                    'Happiness','Sadness','Surprise'],
                                   path_to_save = 'confusion_matrix', name_filename = 'confusion_matrix_L_SVM.pdf',
                                   title='L-SVM')

    # calculate the metric with deleted -1 labels
    metric = 0.67 * f1_score(ground_truth_predictions.values[delete_mask], final_prediction,
                             average='macro') + 0.33 * accuracy_score(ground_truth_predictions.values[delete_mask], final_prediction)
    print('final metric:', metric)
    print('F1-score:',f1_score(ground_truth_predictions.values[delete_mask], final_prediction, average='macro'))
    print('accuracy:', accuracy_score(ground_truth_predictions.values[delete_mask], final_prediction))


if __name__=='__main__':
    main()

