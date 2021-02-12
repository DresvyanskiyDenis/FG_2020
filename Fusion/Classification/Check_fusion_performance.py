import os

import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from Audio.Regression.Preprocessing.labels_utils_regression import construct_video_filename_from_label, \
    get_video_frame_rate, extend_sample_rate_of_labels
from Fusion.Classification.prediction_fusion import align_sample_rate_to_video_rate


def main():
    path_to_elena_predictions='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\predictions\\elena_nn_svm_predictions_classification\\'
    path_to_heysem_predictions='C:\\Users\\Dresvyanskiy\\Downloads\\vggfer_scores\\'
    path_to_pann_predictions='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\Audio\\logs\\Maxim\\'
    path_to_denis_predictions='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\Audio\\logs\\Denis\\'
    path_to_ground_truth='D:\\Databases\\AffWild2\\Annotations\\EXPR_Set\\validation\\Aligned_labels\\'
    path_to_video='D:\\Databases\\AffWild2\\Videos\\'

    labels_filenames=os.listdir(path_to_denis_predictions+'predictions_val\\')
    #path_to_filenames_labels='C:\\Users\\Dresvyanskiy\\Desktop\\expression_test_set.txt'
    #labels_filenames=pd.read_csv(path_to_filenames_labels, header=None).values.reshape((-1))
    elena_predictions=pd.DataFrame()
    pann_predictions=pd.DataFrame()
    denis_predictions=pd.DataFrame()
    heysem_predictions=pd.DataFrame
    ground_truth=pd.DataFrame()
    for lbs_filename in labels_filenames:
        filename=lbs_filename.split('.')[0]
        gt=pd.read_csv(path_to_ground_truth+filename+'.txt', header=None)

        el=pd.read_csv(path_to_elena_predictions+'validation\\'+filename+'.txt')
        el=pd.DataFrame(data=el.iloc[:,1:].values)

        pann=pd.read_csv(path_to_pann_predictions+'predictions_val\\'+filename+'.csv', header=None)
        pann=align_sample_rate_to_video_rate(pann, path_to_video, filename, 5)
        #pann.to_csv('C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\Audio\\logs\\Maxim\\predictions_test\\'+filename+'.csv', index=False, header=False)

        de=pd.read_csv(path_to_denis_predictions+'predictions_val\\'+filename+'.csv', header=None)
        de = align_sample_rate_to_video_rate(de, path_to_video, filename, 5)

        he=pd.read_csv(path_to_heysem_predictions+'vggfer_validation\\'+filename+'.csv', header=None)
        he=align_sample_rate_to_video_rate(he, path_to_video, filename, 5)

        if denis_predictions.shape[0]==0:
            pann_predictions=pann
            elena_predictions=el
            denis_predictions=de
            ground_truth=gt
            heysem_predictions=he
        else:
            pann_predictions=pann_predictions.append(pann)
            elena_predictions=elena_predictions.append(el)
            denis_predictions=denis_predictions.append(de)
            ground_truth=ground_truth.append(gt)
            heysem_predictions=heysem_predictions.append(he)

    predictions=[elena_predictions, denis_predictions, pann_predictions, heysem_predictions]
    num_predictions=len(predictions)
    num_classes=7
    weights=np.loadtxt('C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\FG_2020\\Audio\\logs\\predictions_for_submission\\trial_7\\weights_for_fusion.txt')
    final_prediction=predictions[0]*weights[0]
    for i in range(1, num_predictions):
        final_prediction+=predictions[i]*weights[i]
    final_prediction=np.argmax(final_prediction.values, axis=-1)
    delete_mask=ground_truth.values!=-1

    metric=0.67 * f1_score(final_prediction.reshape((-1,1))[delete_mask], ground_truth.values[delete_mask], average='macro') + 0.33 * accuracy_score(final_prediction.reshape((-1,1))[delete_mask],ground_truth.values[delete_mask])
    print('final metric:',metric)
    print('F1-score:',f1_score(final_prediction.reshape((-1,1))[delete_mask], ground_truth.values[delete_mask], average='macro'))
    print('accuracy:', accuracy_score(final_prediction.reshape((-1,1))[delete_mask],ground_truth.values[delete_mask]))



if __name__ == "__main__":
    main()