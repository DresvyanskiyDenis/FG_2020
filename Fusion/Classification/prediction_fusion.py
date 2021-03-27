import os

import cv2
import pandas as pd
import numpy as np
import scipy
from sklearn.metrics import f1_score, accuracy_score
from scipy.special import softmax

from Audio.Regression.Preprocessing.labels_utils_regression import construct_video_filename_from_label, \
    get_video_frame_rate, extend_sample_rate_of_labels


def align_sample_rate_to_video_rate(predictions, path_to_video, filename, original_sample_rate):
    video_filename = construct_video_filename_from_label(path_to_video=path_to_video,
                                                         label_filename=filename)
    video_frame_rate = get_video_frame_rate(path_to_video + video_filename)

    predictions = extend_sample_rate_of_labels(predictions, original_sample_rate, video_frame_rate)
    predictions = predictions.astype('float32')

    # align to video amount of frames
    cap = cv2.VideoCapture(path_to_video + video_filename)
    video_frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    aligned_predictions = np.zeros(shape=(video_frame_length, predictions.shape[1]), dtype='float32')
    predictions = predictions.values
    if video_frame_length <= predictions.shape[0]:
        aligned_predictions[:] = predictions[:video_frame_length]
    else:
        aligned_predictions[:predictions.shape[0]] = predictions[:]
        value_to_fill = predictions[-1]
        aligned_predictions[predictions.shape[0]:] = value_to_fill
    aligned_predictions = pd.DataFrame(data=aligned_predictions)
    return aligned_predictions


def main():
    path_to_model_3_predictions = 'C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\Pretrained_AffectNet_model_3\\devel'
    path_to_model_4_predictions = 'C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\Pretrained_AffectNet_model_4\\devel'
    path_to_1D_CNN_predictions = 'C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\1D_CNN\\devel'
    path_to_PANN_predictions = 'C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\PANN\\devel'
    path_to_linearSVM_predictions = 'C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\linearSVM_2_1_z_l2_C_0_01\\'
    path_to_ground_truth_predictions = 'E:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels\\'
    path_to_videos = 'E:\\Databases\\AffWild2\\Videos\\'
    num_classes = 7

    # load all filenames with predictions
    labels_filenames = os.listdir(path_to_model_3_predictions)
    # create variables to load predictions
    model_3_predictions = []
    model_4_predictions = []
    model_1D_predictions = []
    model_PANN_predictions = []
    model_linearSVM_predictions = []
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
    model_3_predictions = pd.concat(model_3_predictions)
    model_4_predictions = pd.concat(model_4_predictions)
    model_PANN_predictions = pd.concat(model_PANN_predictions)
    model_1D_predictions = pd.concat(model_1D_predictions)
    model_linearSVM_predictions = pd.concat(model_linearSVM_predictions)
    ground_truth_predictions = pd.concat(ground_truth_predictions)

    predictions=[model_3_predictions, model_4_predictions,model_1D_predictions, model_linearSVM_predictions]
    num_predictions=len(predictions)
    num_weights=10000
    num_classes=7
    weights=np.zeros(shape=(num_weights, num_classes,num_predictions ))
    for i in range(num_weights):
        weights[i]=np.random.dirichlet(alpha=np.ones((num_predictions,)), size=num_classes)

    best=0
    best_weights=None
    for weight_idx in range(num_weights):
        final_prediction=predictions[0]*weights[weight_idx,:,0]
        for i in range(1, num_predictions):
            final_prediction+=predictions[i]*weights[weight_idx,:,i]
        final_prediction=np.argmax(final_prediction.values, axis=-1)
        delete_mask=ground_truth_predictions.values!=-1

        metric=0.67 * f1_score(final_prediction.reshape((-1,1))[delete_mask], ground_truth_predictions.values[delete_mask], average='macro') + \
               0.33 * accuracy_score(final_prediction.reshape((-1,1))[delete_mask],ground_truth_predictions.values[delete_mask])
        #print('current metric:', metric)
        if metric>best:
            best=metric
            best_weights=weights[weight_idx]
            print('best metric now: %f'%(best))
            print('weights:', best_weights)
    print('final best metric:%f'%(best))
    print('weights:', best_weights)

    # generate test predictions
    path_to_save='C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\predictions_for_submission\\'
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    folder_to_save='Elena_AffectNet_3_4_and_1DCNN_LinearSVM\\'
    if not os.path.exists(path_to_save+folder_to_save):
        os.mkdir(path_to_save+folder_to_save)
    np.savetxt(path_to_save+folder_to_save+'weights_for_fusion.txt', best_weights)
    '''folder_to_save_predictions='test_predictions\\'
    if not os.path.exists(path_to_save+folder_to_save+folder_to_save_predictions):
        os.mkdir(path_to_save+folder_to_save+folder_to_save_predictions)'''

    '''columns_for_test='Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise\n'
    #path_to_filenames_test_labels='C:\\Users\\Dresvyanskiy\\Desktop\\expression_test_set.txt'
    #labels_filenames=pd.read_csv(path_to_filenames_test_labels, header=None).values.reshape((-1))
    labels_filenames=os.listdir('C:\\Users\\Denis\\PycharmProjects\\FG_2020\\Predictions\\1D_CNN\\test\\')
    for lbs_filename in labels_filenames:
        filename = lbs_filename.split('.')[0]

        el=pd.read_csv(path_to_elena_predictions+'test\\'+filename+'.txt')
        el=pd.DataFrame(data=el.iloc[:,1:].values)

        #pann=pd.read_csv(path_to_pann_predictions+'predictions_test\\'+filename+'.csv', header=None)
        #pann=align_sample_rate_to_video_rate(pann, path_to_video, filename, 5)

        de=pd.read_csv(path_to_denis_predictions+'test\\'+filename+'.csv', header=None)
        de = align_sample_rate_to_video_rate(de, path_to_video, filename, 5)

        he=pd.read_csv(path_to_heysem_predictions+'test\\'+filename+'.txt')
        #he=align_sample_rate_to_video_rate(he, path_to_video, filename, 5)
        he = pd.DataFrame(data=he.iloc[:, 1:].values)


        predictions=[el, de,he]

        final_test_prediction=predictions[0]*best_weights[:,0]
        for i in range(1, num_predictions):
            final_test_prediction += predictions[i] * best_weights[:,i]
        final_test_prediction = np.argmax(final_test_prediction.values, axis=-1).reshape((-1,1)).astype('int32')
        file=open(path_to_save+folder_to_save+folder_to_save_predictions+filename+'.txt', 'w')
        file.write(columns_for_test)
        file.close()
        pd.DataFrame(data=final_test_prediction).to_csv(path_to_save+folder_to_save+folder_to_save_predictions+filename+'.txt', header=False, index=False, mode='a')
    '''




if __name__ == "__main__":
    main()