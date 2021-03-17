import copy
import gc
from typing import Dict, Tuple, Optional, List, Union
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from VIdeo.SVM_related_classification.utils.normalization_utils import z_normalization, power_normalization
from VIdeo.SVM_related_classification.utils.sequence_utils import cut_data_on_chunks, extract_statistics_from_2d_window

Data_dict_type = Dict[str, Tuple[np.ndarray, int]]
Labels_dict_type = Dict[str, pd.DataFrame]
Labels_dict_type_numpy=Dict[str, np.ndarray]


def load_sample_rates(path: str) -> Dict[str, int]:
    """Loads video sample rates from path.
       The sample rates contain in special file with .txt extention.
       The file format:
       filename, sample_rate
       filename, sample_rate
       ...

    :param path: str
                path to the file with saved sample rates
    :return: Dict[str, int]
                Loaded sample rates in format Dict[filename->sample_rate]
    """
    sample_rates = pd.read_csv(path)
    result_dict = {x['filename'].split('.')[0]: x['frame_rate'] for index, x in sample_rates.iterrows()}
    return result_dict


def get_sample_rate_according_filename(filename: str, sample_rates: Dict[str, int]) -> int:
    """Extract sample rate from sample_rates dict according to provided filename.
       We need this separate function, because some annotation files have "_left" and "_right" suffixes,
       which denote left or right human on the video. However, the for both annotations there is only one sample rate
       (because it is one video). That is why, if filename with suffix "_right" or "_left" comes, the function extracts
       the main filename and use it to get sample rate.
       Simple example:
       "video1_left" -> "video1" -> return sample_rates["video1"]

    :param filename: str
                filename to get sample rate of it. See description.
    :param sample_rates: Dict[str, int]
                sample rates in format Dict[filename->sample_rate].
    :return: int
                extracted sample rate according file of video.
    """
    if filename.split('_')[-1] in ('right', 'left'):
        filename = filename.split('_')[0]
    return sample_rates[filename]


def transform_data_and_labels_to_dict(data: np.ndarray, labels: pd.DataFrame,
                                      sample_rates: Dict[str, int]) -> Tuple[Data_dict_type, Labels_dict_type]:
    """Transforms data from np.ndarray (with features of all videofiles) to Dict[filename->(features, sample_rate)]
    Transforms labels from pd.DataFrame (with labels of all videofiles) to Dict[filename->pd.DataFrame]

    :param data: np.ndarray
                Loaded from .pickle file features of all videofiles (for each frame of each video)
    :param labels: pd.DataFrame
                Loaded from .csv file labels of all videofiles (for each frame of each video)
    :param sample_rates: Dict[str, int]
                sample rates of videofiles in format Dict[filename->sample_rate].
    :return: Tuple[Data_dict_type, Labels_dict_type]
                separated according video filenames data and labels in formats:
                Data_dict_type
                Labels_dict_type
    """
    # create dicts
    data_dict = {}
    labels_dict = {}
    # find unique filenames from entire DataFrame
    paths = labels['name_folder'].unique()
    for path in paths:
        # create mask to get only features and labels according path
        mask = labels['name_folder'] == path
        masked_data = data[mask]
        sample_rate = get_sample_rate_according_filename(path, sample_rates)
        # save extracted information as separate instance of dict
        data_dict[path] = (masked_data, sample_rate)
        labels_dict[path] = labels[mask]
    return data_dict, labels_dict


def cut_all_data_and_labels_on_chunks(data: Data_dict_type, labels: Labels_dict_type,
                                      window_size: float, window_step: float) -> Tuple[
    Data_dict_type, Labels_dict_type]:
    """Cuts data and labels on chunks of defined size.

    :param data: Data_dict_type
                data in format Dict[filename->(features, sample_rate)]
    :param labels: Labels_dict_type
                labels in format Dict[filename->pd.DataFrame]
    :param window_size: float
                size of window (chunk) in seconds
    :param window_step: float
                step of window (chunk) in seconds
    :return: Tuple[Data_dict_type, Labels_dict_type]
                Cut data and labels according to defined window size and step.
    """
    for key, item in data.items():
        # extract data and sample rate of videofile
        data_array, sample_rate = item
        # calculate size of window in units (indexes)
        window_size_in_units = int(np.round(window_size * sample_rate))
        window_step_in_units = int(np.round(window_step * sample_rate))
        try:
            # try to cut data on chunks with defined window
            data_array = cut_data_on_chunks(data_array, window_size_in_units, window_step_in_units)
            data_array = np.concatenate([x[np.newaxis, ...] for x in data_array], axis=0)
        except AttributeError:
            # if size of window or step of window are more than length of data, takes full data as one window.
            data_array = data_array[np.newaxis, ...]
        data[key] = (data_array, sample_rate)
        # labels cutting, everything the same as with data cutting
        labels_dataframe = labels[key]
        try:
            labels_dataframe = cut_data_on_chunks(labels_dataframe.values, window_size_in_units, window_step_in_units)
            labels_dataframe = np.concatenate([x[np.newaxis, ...] for x in labels_dataframe], axis=0)
        except AttributeError:
            # labels now will be saved in np.ndarray format
            labels_dataframe = labels_dataframe.values[np.newaxis, ...]
        labels[key] = labels_dataframe
    return data, labels


def average_labels_within_window(labels: Labels_dict_type_numpy) -> Labels_dict_type_numpy:
    """Averages labels within window to one number (class)

    :param labels: Labels_dict_type_numpy
                Labels in format Dict[filename->labels]. Labels have shape (num_windows, window_size)
    :return: Labels_dict_type_numpy
                Averaged labels in format Dict[filename->labels].Now they have shape (num_windows, 1)
    """
    # do deep copy of labels to eliminate changing original array
    labels_copy = copy.deepcopy(labels)
    for key, item in labels_copy.items():
        labels_windows = item
        # future averaged labels array
        averaged_labels = np.zeros((labels_windows.shape[0], 1))
        for window_idx in range(labels_windows.shape[0]):
            # idx=2 is an emotion label. Mode calculates the most frequent class met in current window.
            # the most frequent = averaging will be shifted in its direction, so, this class will be assign
            # to curernt window
            the_most_frequent = mode(labels_windows[window_idx][:, 2])[0]
            averaged_labels[window_idx] = the_most_frequent
        labels_copy[key] = averaged_labels
    return labels_copy


def delete_instances_with_class(data: Data_dict_type, labels: Labels_dict_type_numpy,
                                class_to_delete: int) -> Tuple[Data_dict_type, Labels_dict_type_numpy]:
    """Deletes windows from data and labels dicts with value=class_to_delete. (for example, samples assigned to -1,
    which means no class).

    :param data: Data_dict_type
                data in format Dict[filename->(features, sample_rate)]
    :param labels: Labels_dict_type_numpy
                labels in format Dict[filename->np.ndarray]
    :param class_to_delete: int
                class needed to be deleted from all instances (rows)
    :return:Tuple[Data_dict_type, Labels_dict_type_numpy]
                data and labels dicts with deleted rows, labels of which were equalled to class_to_delete
    """
    # if dict after deleting samples with class_to_delete labels will be empty, it should be deleted
    keys_to_delete = []
    for key, item in data.items():
        # extract current data and labels arrays
        values, sample_rate = item
        current_labels = labels[key]
        # create mask of items, which will remain
        mask = current_labels[:, 0] != class_to_delete
        # if all elements of mask == False, then drop such instance(videofile) from database
        if mask.any() == False:
            keys_to_delete.append(key)
        else:
            values = values[mask]
            current_labels = current_labels[mask]
            data[key] = (values, sample_rate)
            labels[key] = current_labels
    # delete saved keys (videofiles with 0 rows/windows remained)
    for key in keys_to_delete:
        data.pop(key)
        labels.pop(key)
    return data, labels


def concatenate_all_data_and_labels(data: Data_dict_type, labels: Labels_dict_type) -> Tuple[np.ndarray, np.ndarray]:
    concatenated_data = []
    concatenated_labels = []
    for key, item in data.items():
        values, sample_rate = item
        current_labels = labels[key]
        concatenated_data.append(values)
        concatenated_labels.append(current_labels)
    concatenated_data = np.concatenate(concatenated_data, axis=0)
    concatenated_labels = np.concatenate(concatenated_labels, axis=0)
    return concatenated_data, concatenated_labels


def extract_statistics_from_windows(data: Data_dict_type) -> Data_dict_type:
    for key, item in data.items():
        values, sample_rate = item
        functionals = []
        for window_idx in range(values.shape[0]):
            window_functionals = extract_statistics_from_2d_window(values[window_idx])
            functionals.append(window_functionals[np.newaxis, ...])
        functionals = np.concatenate(functionals, axis=0)
        functionals = functionals.reshape(functionals.shape[:-1])
        data[key] = (functionals, sample_rate)
    return data


def recover_frame_labels_from_window_labels(labels: np.ndarray, original_labels: np.ndarray, sample_rate: int,
                                            window_size: float, window_step: float) -> np.ndarray:
    window_size_in_units = int(np.round(window_size * sample_rate))
    window_step_in_units = int(np.round(window_step * sample_rate))
    if original_labels.shape[0] < window_size_in_units:
        window_size_in_units = original_labels.shape[0]
        window_step_in_units = original_labels.shape[0]
    num_windows = labels.shape[0]
    num_class_probs = labels.shape[-1]
    timesteps = np.array([i for i in range(original_labels.shape[0])])
    cut_timesteps = cut_data_on_chunks(timesteps, window_size_in_units, window_step_in_units)
    cut_timesteps = np.concatenate([x[np.newaxis, ...] for x in cut_timesteps], axis=0)
    expanded_instance = np.ones((num_windows, window_size_in_units, num_class_probs))
    expanded_instance = expanded_instance * labels[:, np.newaxis, :]
    # flatten timesteps and expanded instance
    cut_timesteps = cut_timesteps.reshape((-1, 1))
    expanded_instance = expanded_instance.reshape((-1, num_class_probs))
    # concatenate it in pandas DataFrame and group by timesteps
    column_names = ['timestep'] + ['class_prob_%i' % (i) for i in range(labels.shape[1])]
    grouping_DataFrame = pd.DataFrame(data=np.concatenate([cut_timesteps, expanded_instance], axis=-1),
                                      columns=column_names)
    grouping_DataFrame = grouping_DataFrame.groupby('timestep').mean()
    # check if recovered labels have the same length as original
    if grouping_DataFrame.shape[0] != original_labels.shape[0]:
        raise Exception('The shape of recovered labels does not equal to the original length.')
    # assign recovered labels to labels. Argmax need to take max from sores of each class
    instance = grouping_DataFrame.values.argmax(axis=-1)
    return instance


def recover_labels_in_dict(predictions: dict, sample_rates: dict, original_labels: dict, window_size: float,
                           window_step: float) -> Dict[str, np.ndarray]:
    for key, item in predictions.items():
        current_sample_rate = get_sample_rate_according_filename(key, sample_rates)
        recovered_labels = recover_frame_labels_from_window_labels(item, original_labels[key], current_sample_rate,
                                                                   window_size, window_step)
        predictions[key] = recovered_labels
    return predictions


def get_normalizers_trained_on_dict_data(data: Data_dict_type,
                                         normalizer_types: Tuple[str, ...] = ('z', 'power')) -> Tuple[object, ...]:
    concatenated_data = []
    for key, item in data.items():
        values, sample_rate = item
        concatenated_data.append(values)
    concatenated_data = np.concatenate(concatenated_data, axis=0)
    normalizers = []
    for normalizer_type in normalizer_types:
        if normalizer_type == 'z':
            concatenated_data, z_normalizer = z_normalization(concatenated_data, return_scaler=True)
            normalizers.append(z_normalizer)
        elif normalizer_type == 'power':
            concatenated_data, power_normalizer = power_normalization(concatenated_data, return_scaler=True)
            normalizers.append(power_normalizer)
        elif normalizer_type == 'l2':
            normalizers.append(Normalizer(norm='l2'))
        else:
            raise AttributeError(
                'Normalizer_types is supported only z and power normalization. Got %s.' % (normalizer_types))
    del concatenated_data
    gc.collect()
    return tuple(normalizers)


def normalize_all_data_in_dict(data: Data_dict_type, normalizers: Tuple[object, ...]) -> Data_dict_type:
    for key, item in data.items():
        values, sample_rate = item
        old_shape = values.shape
        values = values.reshape((-1, values.shape[-1]))
        for normalizer in normalizers:
            values = normalizer.transform(values)
        values = values.reshape(old_shape)
        data[key] = (values, sample_rate)
    return data


def prepare_data_and_labels_for_svm(data: Data_dict_type, labels: Labels_dict_type, window_size: float,
                                    window_step: float,
                                    normalization: bool = False, normalization_types: Tuple[str, ...] = ('z', 'l2'),
                                    return_normalizers: bool = False,
                                    normalizers: Optional[Tuple[object, ...]] = None,
                                    class_to_delete: Optional[int] = None) \
        -> Union[
            Tuple[Data_dict_type, Labels_dict_type],
            Tuple[Data_dict_type, Labels_dict_type, Optional[Tuple[object, ...]]]
        ]:
    # cut data on sequences
    data, labels = cut_all_data_and_labels_on_chunks(data, labels, window_size, window_step)
    labels_averaged = average_labels_within_window(labels)

    # delete instances with -1 label
    if not class_to_delete is None:
        data, labels_averaged = delete_instances_with_class(data, labels_averaged, class_to_delete)
    # extract window statistics such as mean, std
    data = extract_statistics_from_windows(data)
    # get normalizers of data
    if normalization:
        if normalizers is None:
            normalizers = get_normalizers_trained_on_dict_data(data, normalization_types)
        data = normalize_all_data_in_dict(data, normalizers)
    if return_normalizers: return data, labels_averaged, normalizers
    return data, labels_averaged


def validate_estimator_on_dict(estimator: object, val_data: Data_dict_type, sample_rates: Dict[str, int],
                               original_labels: Labels_dict_type,
                               window_size: float, window_step: float) -> float:
    # generate predictions
    predictions = dict()
    for key, item in val_data.items():
        values, sample_rate = item
        num_windows = values.shape[0]
        values = values.reshape((-1, values.shape[-1]))
        predictions[key] = estimator.decision_function(values)
    # recover frames
    predictions = recover_labels_in_dict(predictions=predictions, sample_rates=sample_rates,
                                         original_labels=original_labels, window_size=window_size,
                                         window_step=window_step)
    # concatenate predictions to one sequene
    concat_predictions = []
    concat_ground_truth_labels = []
    for key, item in predictions.items():
        concat_predictions.append(item)
        concat_ground_truth_labels.append(original_labels[key].iloc[:, 2])
    concat_predictions = np.concatenate(concat_predictions, axis=0)
    concat_ground_truth_labels = np.concatenate(concat_ground_truth_labels, axis=0)
    # delete -1 labels
    mask = concat_ground_truth_labels != -1
    concat_predictions = concat_predictions[mask]
    concat_ground_truth_labels = concat_ground_truth_labels[mask]
    return 0.33 * accuracy_score(concat_ground_truth_labels, concat_predictions) \
           + 0.67 * f1_score(concat_ground_truth_labels, concat_predictions, average='macro')


def main():
    load_path_train_data = 'D:\\Downloads\\aff_wild2_train_emo_with_loss.pickle'
    load_path_train_labels = 'D:\\Downloads\\df_affwild2_train_emo.csv'
    load_path_val_data = 'D:\\Downloads\\aff_wild2_val_emo_with_loss.pickle'
    load_path_val_labels = 'D:\\Downloads\\df_affwild2_val_emo.csv'
    load_path_sample_rates = 'D:\\Downloads\\videos_frame_rate.txt'
    window_size = 4.
    window_step = 2.
    normalization_types = ('z', 'l2')
    # load data, labels and sample rates
    sample_rates = load_sample_rates(load_path_sample_rates)
    train_data = np.load(load_path_train_data, allow_pickle=True)
    train_labels = pd.read_csv(load_path_train_labels)

    # make first column to be str type (some instances incorrectly defined as int or float values)
    train_labels['name_folder'] = train_labels['name_folder'].astype('str')
    # transform it to convenient format (in dict[filename->data])
    train_data, train_labels = transform_data_and_labels_to_dict(train_data, train_labels, sample_rates)
    # preprocess training data
    prepared_train_data, prepared_train_labels, train_normalizers = prepare_data_and_labels_for_svm(
        data=train_data, labels=train_labels, window_size=window_size, window_step=window_step,
        normalization=True, normalization_types=normalization_types, return_normalizers=True,
        normalizers=None, class_to_delete=-1)
    # clear RAM
    gc.collect()
    # all the same operations for validation data
    # load val data
    val_data = np.load(load_path_val_data, allow_pickle=True)
    val_labels = pd.read_csv(load_path_val_labels)
    val_labels['name_folder'] = val_labels['name_folder'].astype('str')
    val_data, val_labels = transform_data_and_labels_to_dict(val_data, val_labels, sample_rates)
    val_data, _ = prepare_data_and_labels_for_svm(
        data=val_data, labels=val_labels.copy(), window_size=window_size, window_step=window_step,
        normalization=True, normalization_types=normalization_types, return_normalizers=False,
        normalizers=train_normalizers, class_to_delete=None)

    # clear RAM
    gc.collect()
    # train SVM
    prepared_train_data, prepared_train_labels = concatenate_all_data_and_labels(prepared_train_data,
                                                                                 prepared_train_labels)
    # prepare labels to fit it in SVC
    prepared_train_labels = prepared_train_labels.reshape((-1,))
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        linearSVM = LinearSVC(C=C)
        linearSVM = linearSVM.fit(prepared_train_data, prepared_train_labels)
        # validation with recovering the number of frames of each video
        score = validate_estimator_on_dict(estimator=linearSVM, val_data=val_data, sample_rates=sample_rates,
                                           original_labels=val_labels,
                                           window_size=window_size, window_step=window_step)
        print('C:', C, 'score:', score)


if __name__ == '__main__':
    main()
