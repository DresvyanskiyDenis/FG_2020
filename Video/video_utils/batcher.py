import os
import pandas as pd
import numpy as np
import gc

from Audio.utils.utils import CCC_2_sequences_numpy

mean=(91.4953, 103.8827, 131.0912)

def frames_batch_generator(path_to_batches, batch_size, data_prefix='data_batch_num_', labels_prefix='labels_timesteps_batch_num_', labels_type=[]):
    batch_filenames=os.listdir(path_to_batches)
    num_batches=int(len(batch_filenames)/2)
    batches_idx_permutations=np.random.permutation(num_batches)
    tmp_data_window=np.load(path_to_batches+batch_filenames[0])
    batch_index=0
    for i in range(0, num_batches, batch_size):
        if i+batch_size>num_batches:
            current_batch_size=num_batches-i
        else:
            current_batch_size=batch_size
        data_batches=[]
        labels_timesteps_batches=[]
        for idx_batch in range(current_batch_size):
            data_batches.append(np.load(path_to_batches + data_prefix + str(batches_idx_permutations[batch_index]) + '.npy').astype('float32'))
            labels_timesteps_batches.append(np.load(path_to_batches+ labels_prefix+str(batches_idx_permutations[batch_index])+'.npy').astype('float32'))
            batch_index+=1
        data=np.vstack(data_batches)
        data=data[...,::-1]-mean
        if len(labels_type)==2:
            labels=np.vstack(labels_timesteps_batches)[...,:2]
        else:
            if labels_type[0]=='valence':
                labels = np.vstack(labels_timesteps_batches)[..., 0]
            elif labels_type[0]=='arousal':
                labels = np.vstack(labels_timesteps_batches)[..., 1]
        timesteps=np.vstack(labels_timesteps_batches)[...,-1]
        yield data, labels, timesteps

def calculate_model_performance_by_path_to_validation_batches(model, path_to_batches, label_type=[], delete_value=-1):
    #TODO: change it, because you changed forming the validation batches
    filenames=os.listdir(path_to_batches)
    data_filenames=[value for value in filenames if 'data' in value]
    data_filenames=sorted(data_filenames)
    labels_timesteps_filenames=[value for value in filenames if 'labels' in value]
    labels_timesteps_filenames = sorted(labels_timesteps_filenames)
    ground_truth_predictions=pd.DataFrame()
    label_type=['prediction_'+str(i) for i in label_type]
    for file_idx in range(len(data_filenames)):
        data=np.load(path_to_batches+data_filenames[file_idx]).astype('float32')
        data = data[..., ::-1] - mean
        num_windows=data.shape[0]
        window_size=data.shape[1]
        labels_timesteps=pd.read_csv(path_to_batches+labels_timesteps_filenames[file_idx])
        for i in range(len(label_type)):
            labels_timesteps[label_type[i]]=np.NaN
        predictions=model.predict(data, batch_size=1)
        num_labels = len(label_type)
        for window_idx in range(num_windows):
            labels_timesteps.iloc[window_idx*window_size:window_idx*window_size+window_size, -num_labels:]=predictions[window_idx,:]
        labels_timesteps=labels_timesteps.groupby(by=['timestep']).mean()
        if ground_truth_predictions.shape[0]==0:
            ground_truth_predictions=labels_timesteps
        else:
            ground_truth_predictions=ground_truth_predictions.append(labels_timesteps)
        del data, labels_timesteps
        gc.collect()
    results=[]
    mask_to_delete=ground_truth_predictions.index!=delete_value
    ground_truth_predictions=ground_truth_predictions[mask_to_delete]
    for i in range(len(label_type)):
        result=CCC_2_sequences_numpy(y_true=ground_truth_predictions[label_type[i].split('prediction_')[-1]].values,
                                     y_pred=ground_truth_predictions[label_type[i]].values)
        results.append(result)
    return results

