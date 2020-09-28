import os

import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

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
    path_to_elena_predictions='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\predictions\\elena_nn_svm_predictions_classification\\'
    path_to_pann_predictions='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\predictions\\pann_predictions_classification\\'
    path_to_denis_predictions='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\predictions\\prediction_probabilities_extended_aligned\\'
    path_to_ground_truth='D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels\\'
    path_to_video='D:\\Databases\\AffWild2\\Videos\\'
    labels_filenames=os.listdir(path_to_denis_predictions+'validation\\')
    elena_predictions=pd.DataFrame()
    #pann_predictions=pd.DataFrame()
    denis_predictions=pd.DataFrame()
    ground_truth=pd.DataFrame()
    for lbs_filename in labels_filenames:
        filename=lbs_filename.split('.')[0]
        gt=pd.read_csv(path_to_ground_truth+filename+'.txt', header=None)

        el=pd.read_csv(path_to_elena_predictions+'validation\\'+filename+'.txt')
        el=pd.DataFrame(data=el.iloc[:,1:].values)

        #pann=pd.read_csv(path_to_pann_predictions+'validation\\'+filename+'.csv', header=None)
        #pann=align_sample_rate_to_video_rate(pann, path_to_video, filename, 5)

        de=pd.read_csv(path_to_denis_predictions+'validation\\'+filename+'.csv', header=None)
        de = align_sample_rate_to_video_rate(de, path_to_video, filename, 5)

        if denis_predictions.shape[0]==0:
            #pann_predictions=pann
            elena_predictions=el
            denis_predictions=de
            ground_truth=gt
        else:
            #pann_predictions=pann_predictions.append(pann)
            elena_predictions=elena_predictions.append(el)
            denis_predictions=denis_predictions.append(de)
            ground_truth=ground_truth.append(gt)

    predictions=[ denis_predictions, elena_predictions]
    num_predictions=len(predictions)
    num_weights=10000
    weights=np.zeros(shape=(num_weights, num_predictions ))
    for i in range(num_weights):
        weights[i]=np.random.dirichlet(alpha=[1 for i in range(num_predictions)], size=1)

    best=0
    best_weights=None
    for weight_idx in range(num_weights):
        final_prediction=predictions[0]*weights[weight_idx, 0]
        for i in range(1, num_predictions):
            final_prediction+=predictions[i]*weights[weight_idx, i]
        final_prediction=np.argmax(final_prediction.values, axis=-1)
        delete_mask=ground_truth.values!=-1

        metric=0.67 * f1_score(final_prediction.reshape((-1,1))[delete_mask], ground_truth.values[delete_mask], average='macro') + 0.33 * accuracy_score(final_prediction.reshape((-1,1))[delete_mask],ground_truth.values[delete_mask])
        if metric>best:
            best=metric
            best_weights=weights[weight_idx]
            print('best metric now: %f'%(best))
    print('final best metric:%f'%(best))
    print('weights:', best_weights)





if __name__ == "__main__":
    main()