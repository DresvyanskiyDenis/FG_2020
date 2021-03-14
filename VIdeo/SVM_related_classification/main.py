import copy
import gc
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import mode

from VIdeo.SVM_related_classification.utils.normalization_utils import z_normalization, power_normalization, \
    l2_normalization
from VIdeo.SVM_related_classification.utils.sequence_utils import cut_data_on_chunks

Data_dict_type = Dict[str, Tuple[np.ndarray, int]]
Labels_dict_type = Dict[str, pd.DataFrame]


def load_sample_rates(path: str) -> Dict[str, int]:
    sample_rates = pd.read_csv(path)
    result_dict = {x['filename'].split('.')[0]: x['frame_rate'] for index, x in sample_rates.iterrows()}
    return result_dict


def get_sample_rate_according_filename(filename: str, sample_rates: Dict[str, int]) -> int:
    if filename.split('_')[-1] in ('right', 'left'):
        filename = filename.split('_')[0]
    return sample_rates[filename]


def transform_data_and_labels_to_dict(data: np.ndarray, labels: pd.DataFrame,
                                      sample_rates: Dict[str, int]) -> Tuple[Data_dict_type, Labels_dict_type]:
    data_dict = {}
    labels_dict = {}
    paths = labels['name_folder'].unique()
    for path in paths:
        mask = labels['name_folder'] == path
        masked_data = data[mask]
        sample_rate = get_sample_rate_according_filename(path, sample_rates)
        data_dict[path] = (masked_data, sample_rate)
        labels_dict[path] = labels[mask]
    return data_dict, labels_dict

def cut_all_data_and_labels_on_chunks(data:Data_dict_type, labels:Labels_dict_type,
                                      window_size:float, window_step:float)->Tuple[Data_dict_type, Labels_dict_type]:
    for key, item in data.items():
        data_array, sample_rate = item
        window_size_in_units=int(np.round(window_size*sample_rate))
        window_step_in_units=int(np.round(window_step*sample_rate))
        data_array=cut_data_on_chunks(data_array, window_size_in_units, window_step_in_units)
        data_array=np.concatenate([x[np.newaxis,...] for x in data_array], axis=0)
        data[key]=(data_array, sample_rate)
        # labels cutting
        labels_dataframe=labels[key]
        labels_dataframe=cut_data_on_chunks(labels_dataframe.values, window_size_in_units, window_step_in_units)
        labels_dataframe = np.concatenate([x[np.newaxis, ...] for x in labels_dataframe], axis=0)
        labels[key]=labels_dataframe
    return data, labels

def average_labels_within_window(labels:Labels_dict_type)->Labels_dict_type:
    labels_copy=copy.deepcopy(labels)
    for key, item in labels_copy.items():
        labels_windows=item
        averaged_labels=np.zeros((labels_windows.shape[0],1))
        for window_idx in range(labels_windows.shape[0]):
            # idx=2 is an emotion label
            the_most_frequent=mode(labels_windows[window_idx][:,2])[0]
            averaged_labels[window_idx]=the_most_frequent
        labels_copy[key]=averaged_labels
    return labels_copy

def normalize_data(data:Data_dict_type, normalization_types:Tuple[str,...]=('z','l2'),
                   return_scalers:bool=False)->Data_dict_type:
    for key, item in data.items():
        values, sample_rate = item
        # iterate through windows to apply normalization independently within window
        for window_idx in range(values.shape[0]):
            window=values[window_idx]
            # apply all provided in normalization_types normalizations
            for normalization_type in normalization_types:
                if normalization_type=='z':
                    window=z_normalization(window)
                elif normalization_type=='power_norm':
                    window=power_normalization(window)
                elif normalization_type=='l2':
                    window=l2_normalization(window)
            values[window_idx]=window
        data[key]=(values, sample_rate)
    return data



def delete_instances_with_class(data:Data_dict_type, labels:Labels_dict_type,
                                class_to_delete:int)->Tuple[Data_dict_type, Labels_dict_type]:
    for key, item in data.items():
        values, sample_rate = item
        current_labels=labels[key]
        mask=current_labels[:,0]!=class_to_delete
        values=values[mask]
        current_labels=current_labels[mask]
        data[key]=(values, sample_rate)
        labels[key]=current_labels
    return data, labels


def concatenate_all_data_and_labels(data:Data_dict_type, labels:Labels_dict_type)-> Tuple[np.ndarray, np.ndarray]:
    concatenated_data=[]
    concatenated_labels=[]
    for key, item in data.items():
        values, sample_rate = item
        current_labels=labels[key]
        concatenated_data.append(values)
        concatenated_labels.append(current_labels)
    concatenated_data=np.concatenate(concatenated_data, axis=0)
    concatenated_labels = np.concatenate(concatenated_labels, axis=0)
    return concatenated_data, concatenated_labels



def main():
    load_path_train_data = 'D:\\Downloads\\aff_wild2_val_emo_with_loss.pickle'
    load_path_train_labels = 'D:\\Downloads\\df_affwild2_val_emo.csv'
    load_path_val_data = 'D:\\Downloads\\aff_wild2_train_emo_with_loss.pickle'
    load_path_val_labels = 'D:\\Downloads\\df_affwild2_train_emo.csv'
    load_path_sample_rates = 'D:\\Downloads\\videos_frame_rate.txt'
    window_size=4.
    window_step=2.
    # load data, labels and sample rates
    sample_rates = load_sample_rates(load_path_sample_rates)
    train_data=np.load(load_path_train_data, allow_pickle=True)
    train_labels=pd.read_csv(load_path_train_labels)

    #val_data = np.load(load_path_val_data, allow_pickle=True)
    #val_labels = pd.read_csv(load_path_val_labels)

    # preprocessing
    train_labels['name_folder'] = train_labels['name_folder'].astype('str')
    #val_labels['name_folder']=val_labels['name_folder'].astype('str')

    # transform it to convenient format (in dict[filename->data])
    train_data, train_labels = transform_data_and_labels_to_dict(train_data, train_labels, sample_rates)
    #val_data, val_labels = transform_data_and_labels_to_dict(val_data, val_labels, sample_rates)

    # cut data on sequences
    train_data, train_labels = cut_all_data_and_labels_on_chunks(train_data, train_labels,window_size, window_step)
    train_labels_averaged=average_labels_within_window(train_labels)
    train_data=normalize_data(train_data)

    #val_data, val_labels = cut_all_data_and_labels_on_chunks(val_data, val_labels,window_size, window_step)
    #val_data=normalize_data(val_data)

    # delete instances with -1 label
    train_data, train_labels_averaged=delete_instances_with_class(train_data, train_labels_averaged, -1)

    # concatenate train data to train SVM
    train_data, train_labels_averaged = concatenate_all_data_and_labels(train_data, train_labels_averaged)
    # clear RAM
    gc.collect()
    a=1+2


if __name__ == '__main__':
    main()
