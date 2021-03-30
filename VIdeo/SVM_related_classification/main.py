import copy
import gc
import os
from typing import Dict, Tuple, Optional, List, Union
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
import seaborn as sns

from VIdeo.SVM_related_classification.utils.normalization_utils import z_normalization, power_normalization
from VIdeo.SVM_related_classification.utils.sequence_utils import cut_data_on_chunks, extract_statistics_from_2d_window

Data_dict_type = Dict[str, Tuple[np.ndarray, int]]
Labels_dict_type = Dict[str, pd.DataFrame]
Labels_dict_type_numpy = Dict[str, np.ndarray]


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
    """Transforms data and labels in dict formats to np.ndarrays, which are concatenated data from all instances
       of these dicts.

    :param data: Dict[str, Tuple[np.ndarray, int]]
            data in format Dict[filename->(values, sample_rate)]
    :param labels: Dict[str, pd.DataFrame]
            labels in format Dict[filename->values]
    :return: Tuple[np.ndarray, np.ndarray]
            two np.ndarrays, which are concatenated data and labels
    """
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
    """Extracts statistics (functionals) across 2-d windows in data (Dict[str, Tuple[np.ndarray, int]])
       Data has shape (num_windows, window_size, num_features)
       returns data with shape (num_windows, new_num_features), where new_num_features = num_features*num_functionals

    :param data: Dict[str, Tuple[np.ndarray, int]]
            data in format Dict[filename->(values, sample_rate)]
    :return: Dict[str, Tuple[np.ndarray, int]]

    """
    for key, item in data.items():
        values, sample_rate = item
        functionals = []
        for window_idx in range(values.shape[0]):
            window_functionals = extract_statistics_from_2d_window(values[window_idx])
            functionals.append(window_functionals[np.newaxis, ...])
        functionals = np.concatenate(functionals, axis=0)
        # squeeze last dimension
        if functionals.shape[-1] == 1:
            functionals = functionals.reshape(functionals.shape[:-1])
        data[key] = (functionals, sample_rate)
    return data


def recover_frame_labels_from_window_labels(labels: np.ndarray, original_labels: np.ndarray, sample_rate: int,
                                            window_size: float, window_step: float) -> np.ndarray:
    """Recover labels in window format to frame format. This means, that labels will be transformed from
       (num_windows,1) to (num_frames,), where num_frames - original number of frames in labels (which correspond to
       number of frames in video file).


    :param labels: np.ndarray
            labels in window format with shape (num_windows, 1)
    :param original_labels: np.ndarray
            original, non-preprocessed labels in frame format with shape (num_frames,)
    :param sample_rate: int
            sample rate of video related to provided original_labels
    :param window_size: float
            the size of window, on which labels were cut, in seconds.
    :param window_step: float
            the step of window, on which labels were cut, in seconds.
    :return: np.ndarray
            transformed to frame format labels
    """
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
    # assign recovered labels to labels.
    instance = grouping_DataFrame.values
    return instance


def recover_labels_in_dict(predictions: dict, sample_rates: dict, original_labels: dict, window_size: float,
                           window_step: float) -> Dict[str, np.ndarray]:
    """Recovers predictions (labels) in window format to frame format. This means, that labels will be transformed from
       (num_windows,1) to (num_frames,), where num_frames - original number of frames in labels (which correspond to
       number of frames in video file).
       The predictions, as well as original_labels, are provided in dict format Dict[filename->values]

    :param predictions: Dict[str, np.ndarray]
            predictions generated by classificator on each window. It has Dict[str, np.ndarray] format, where
            np.ndarray has shape (num_windows,1)
    :param sample_rates: Dict[filename, int]
            Dictionary of video sample rates in format Dict[filename->sample_rate]
    :param original_labels: Dict[str, np.ndarray]
            Original labels in format Dict[filename->np.ndarray], where np.ndarray has shape (num_frames,)
    :param window_size: float
            the size of window, on which labels were cut, in seconds.
    :param window_step: float
            the step of window, on which labels were cut, in seconds.
    :return: Dict[str, np.ndarray]
            transformed labels in format Dict[filename->values], where values have shape (num_frames,)
    """
    for key, item in predictions.items():
        current_sample_rate = get_sample_rate_according_filename(key, sample_rates)
        recovered_labels = recover_frame_labels_from_window_labels(item, original_labels[key], current_sample_rate,
                                                                   window_size, window_step)
        predictions[key] = recovered_labels
    return predictions


def get_normalizers_trained_on_dict_data(data: Data_dict_type,
                                         normalizer_types: Tuple[str, ...] = ('z', 'power')) -> Tuple[object, ...]:
    """Creates and fits defined normalizers on provided data.

    :param data: Dict[str, Tuple[np.ndarray, int]]
            data in format Dict[filename->(values, sample_rate)], where values have
            shape (num_windows, window_size, num_features)
    :param normalizer_types: Tuple[str,...]
            types of normalizers, which will be created and fit on provided data
    :return: Tuple[object,...]
            fit normalizers on provided data
    """
    # concatenate all data from different instances of dict
    concatenated_data = []
    for key, item in data.items():
        values, sample_rate = item
        concatenated_data.append(values)
    concatenated_data = np.concatenate(concatenated_data, axis=0)
    # create and fit normalizers
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
    # clear RAM
    del concatenated_data
    gc.collect()
    return tuple(normalizers)


def normalize_all_data_in_dict(data: Data_dict_type, normalizers: Tuple[object, ...]) -> Data_dict_type:
    """Normalizes data in format Dict[str, Tuple[np.ndarray, int]] with provided normalizers.


    :param data: Dict[str, Tuple[np.ndarray, int]]
            data in format Dict[filename->(values, sample_rate)], where values have
            shape (num_windows, window_size, num_features)
    :param normalizers:
    :return: Dict[str, Tuple[np.ndarray, int]]
            normalized data
    """
    for key, item in data.items():
        values, sample_rate = item
        # save old shape and reshape data to supported format for normalizer
        old_shape = values.shape
        values = values.reshape((-1, values.shape[-1]))
        # normalize data
        for normalizer in normalizers:
            values = normalizer.transform(values)
        # Reshape data back to old shape
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
            Tuple[Data_dict_type, Labels_dict_type_numpy],
            Tuple[Data_dict_type, Labels_dict_type_numpy, Optional[Tuple[object, ...]]]
        ]:
    """Does all preparation of data before feeeding it into classifier.

    :param data: Dict[str, Tuple[np.ndarray, int]]
            data in format Dict[filename->(values, sample_rate)]
    :param labels: Dict[str, pd.DataFrame]
            labels in format Dict[filename->values]
    :param window_size: float
            the size of window, on which labels were cut, in seconds.
    :param window_step: float
            the step of window, on which labels were cut, in seconds.
    :param normalization: bool
            specifies, should ata be normalized or not.
    :param normalization_types: Tuple[str,...]
            if normalization==True, specifies, which normalizations should be applied
    :param return_normalizers: bool
            if normalization=True, specified, should function return fit normalizers, or not
    :param normalizers: Tuple[object,...]
            if normalization=True, provideed list of normalizers, which will transform data
            In that case normalization_types is ignored.
    :param class_to_delete: int
            specifies, should some class be deleted from labels (and data, accordingly)
    :return: Union[
            Tuple[Data_dict_type, Labels_dict_type_numpy],
            Tuple[Data_dict_type, Labels_dict_type_numpy, Optional[Tuple[object, ...]]]]
            returns data and labels in dict format,
            or, if return_normalizers=True: data, labels and List[normalizers]
    """
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
                               window_size: float, window_step: float,
                               plot_and_save_conf_matrix: bool = False) -> Union[float, Tuple[float, np.ndarray]]:
    """Validates provided estimator (classfier) on val_data in format Dict[str, Tuple[np.ndarray, int]]
    Uses recover_labels_in_dict() function to transform features with shape (num_windows, 1) to frame format with
    shape (num_frames,), where num_frames - the number of frames of related video file.

    :param estimator: object
            classifier from sklearn module.
    :param val_data: Dict[str, Tuple[np.ndarray, int]]
            Validation data in format Dict[filename->(values, sample_rate)]
    :param sample_rates: Dict[filename, int]
            Dictionary of video sample rates in format Dict[filename->sample_rate]
    :param original_labels: Dict[str, np.ndarray]
            Original labels in format Dict[filename->np.ndarray], where np.ndarray has shape (num_frames,)
    :param window_size: float
            the size of window, on which labels were cut, in seconds.
    :param window_step: float
            the step of window, on which labels were cut, in seconds.
    :param plot_and_save_conf_matrix: bool
            if True, plots and saves confusion matrix
    :return: float or (float, np.ndarray)
            the estimator's UAR (unweighted average recall)
            or
            the estimator's UAR (unweighted average recall) and confusion matrix with shape (n_classes, n_classes)
    """
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
    # concatenate predictions to one sequence
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
    metric = 0.33 * accuracy_score(concat_ground_truth_labels, concat_predictions.argmax(axis=-1)) \
             + 0.67 * f1_score(concat_ground_truth_labels, concat_predictions.argmax(axis=-1), average='macro')
    if plot_and_save_conf_matrix:
        plot_and_save_confusion_matrix(concat_ground_truth_labels, concat_predictions.argmax(axis=-1),
                                       name_labels=['Neutral', 'Anger', 'Disgust', 'Fear',
                                                    'Happiness', 'Sadness', 'Surprise'],
                                       path_to_save='conf_matrix_C_%s' % estimator.C)
    return metric


def change_class_order_from_AffectNet_to_AffWild2(values: np.ndarray) -> np.ndarray:
    #                              0       1          2           3          4         5         6
    # AffectNet class ordering: Neutral, Happiness, Sadness, Surprised,     Fear,    Disgust, Anger
    # AffWIld2 class ordering : Neutral, Anger,    Disgust,     Fear,    Happiness, Sadness, Surprised
    # so, we need mapping from AffectNet class ordering to AffWild2
    # 0 -> 0
    # 1 -> 6
    # 2 -> 5
    # 3 -> 4
    # 4 -> 1
    # 5 -> 2
    # 6 -> 3
    mapping_dict = {
        -1:-1,
        0: 0,
        1: 4,
        2: 5,
        3: 6,
        4: 3,
        5: 2,
        6: 1
    }

    map_values = np.vectorize(mapping_dict.get)(values)
    return map_values


def save_features_to_file(path: str, features: Data_dict_type, labels: Labels_dict_type_numpy):
    """Saves provided in Dict[str, Tuple[np.ndarray, int]] features with labels in format Dict[str, np.ndarray].
    The provided data and labels will be separated according to filenames (keys of dictionary).
    The saved files will be in .csv format with following columns:
    window_idx, feature_1, feature_2, ... , label

    :param path: str
            path to save features with labels
    :param features: Dict[str, Tuple[np.ndarray, int]]
            features needed to save
    :param labels: Dict[str, np.ndarray]
            labels needed to save
    :return: None
    """
    for key, item in features.items():
        filename = key
        values, sample_rate = item
        window_labels = labels[filename].reshape((-1, 1))
        concatenated_data = np.concatenate(
            [np.array([i for i in range(values.shape[0])])[..., np.newaxis],  # window_idx
             values,  # features
             window_labels], axis=-1)  # labels
        df_to_save = pd.DataFrame(data=concatenated_data)
        columns = ['window_idx'] + ['feature_%i' % i for i in range(values.shape[-1])] + ['label']
        df_to_save.columns = columns
        df_to_save.to_csv(os.path.join(path, filename.split('.')[0] + '.csv'), index=False)


def save_recovered_predictions_to_dir(path_to_save: str, estimator: object,
                                      val_data: Data_dict_type, sample_rates: Dict[str, int],
                                      original_labels: Labels_dict_type,
                                      window_size: float, window_step: float) -> None:
    # TODO: make description
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
    # check if directory exists
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    # save predictions
    num_classes=7
    for key, item in predictions.items():
        pd.DataFrame(data=item, columns=['class_%i_prob'%i for i in range(num_classes)]).to_csv(
            os.path.join(path_to_save, key + '.csv'), index=False)


def visualize_features_according_class(features: np.array, labels: np.array):
    """visualised n-dimensional data with the help of t-SNE algorithm.
    The implementation is taken from sklearn.
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html


    :param features: np.ndarray
            features with shape (n_samples, num_features). They will be reduced to (n_samples, 2)
    :param labels: np.ndarray
            labels of each sample with shape either (n_samples,) or (n_samples, 1)
    :return: None
    """
    # check if labels and features formats are correct
    if len(features.shape) != 2:
        raise AttributeError('Provided features must be 2-dimensional. Got %i.' % len(features.shape))
    if len(labels.shape) > 2:
        raise AttributeError('Provided labels must be 2- or 1-dimensional. Got %i.' % len(labels.shape))
    # reshape labels if they are 2-dimensional
    if len(labels.shape) == 2:
        labels = labels.reshape((-1,))
    # transform data via TSNE
    tsne = TSNE(n_components=2)
    features = tsne.fit_transform(features)
    # create support variables to create graph
    num_classes = np.unique(labels).shape[0]
    colors = [i for i in range(num_classes)]
    class_names = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    # creating graph
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, c, label in zip(range(num_classes), colors, class_names):
        plt.scatter(features[labels == i, 0], features[labels == i, 1], c=c, label=label)
    plt.legend()
    plt.show()


def plot_and_save_confusion_matrix(y_true, y_pred, name_labels, path_to_save: str = 'confusion_matrix',
                                   name_filename: str = 'cm.png'):
    c_m = confusion_matrix(y_true, y_pred)
    conf_matrix = pd.DataFrame(c_m, name_labels, name_labels)

    plt.figure(figsize=(9, 9))

    group_counts = ['{0:0.0f}'.format(value) for value in
                    conf_matrix.values.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         conf_matrix.div(np.sum(conf_matrix, axis=1), axis=0).values.flatten()]

    labels = ['{}\n{}'.format(v1, v2) for v1, v2 in zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(c_m.shape)

    chart = sns.heatmap(conf_matrix,
                        cbar=False,
                        annot=labels,
                        square=True,
                        fmt='',
                        annot_kws={'size': 14},
                        cmap="Blues"
                        )
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    plt.savefig(os.path.join(path_to_save, name_filename), bbox_inches='tight', pad_inches=0)


def main(C_params:List[float], window_size: float = 4, window_step: float = 2, normalization_types: Tuple[str, ...] = ('z', 'l2'),
         generate_test_predictions:bool=False, path_to_save_test_predictions:str='predictions'):
    load_path_train_data = 'D:\\Downloads\\aff_wild2_train_emo_with_loss.pickle'
    load_path_train_labels = 'D:\\Downloads\\df_affwild2_train_emo.csv'
    load_path_val_data = 'D:\\Downloads\\aff_wild2_val_emo_with_loss.pickle'
    load_path_val_labels = 'D:\\Downloads\\df_affwild2_val_emo.csv'
    load_path_test_data = 'D:\\Downloads\\aff_wild2_test_emo_with_loss.pickle'
    load_path_test_labels = 'D:\\Downloads\\df_affwild2_test_emo.csv'
    load_path_sample_rates = 'D:\\Downloads\\videos_frame_rate.txt'
    path_to_save_predictions = 'val_predictions_linearSVM'
    # load data, labels and sample rates
    sample_rates = load_sample_rates(load_path_sample_rates)
    train_data = np.load(load_path_train_data, allow_pickle=True)
    train_labels = pd.read_csv(load_path_train_labels)
    # concatenate train and val to train system on it and make test predictions
    val_data = np.load(load_path_val_data, allow_pickle=True)
    val_labels=pd.read_csv(load_path_val_labels)
    train_data=np.concatenate([train_data, val_data], axis=0)
    train_labels=pd.concat([train_labels, val_labels], axis=0)
    # change the class ordering to AffWild2
    train_labels.iloc[:, 2] = change_class_order_from_AffectNet_to_AffWild2(train_labels.iloc[:, 2])
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
    """# load val data
    val_data = np.load(load_path_val_data, allow_pickle=True)
    val_labels = pd.read_csv(load_path_val_labels)
    # change the class ordering to AffWild2
    val_labels.iloc[:, 2] = change_class_order_from_AffectNet_to_AffWild2(val_labels.iloc[:, 2])
    val_labels['name_folder'] = val_labels['name_folder'].astype('str')
    val_data, val_labels = transform_data_and_labels_to_dict(val_data, val_labels, sample_rates)
    val_data, _ = prepare_data_and_labels_for_svm(
        data=val_data, labels=val_labels.copy(), window_size=window_size, window_step=window_step,
        normalization=True, normalization_types=normalization_types, return_normalizers=False,
        normalizers=train_normalizers, class_to_delete=None)"""

    # clear RAM
    gc.collect()
    # train SVM
    prepared_train_data, prepared_train_labels = concatenate_all_data_and_labels(prepared_train_data,
                                                                                 prepared_train_labels)
    # prepare labels to fit it in SVC
    prepared_train_labels = prepared_train_labels.reshape((-1,))
    for C in C_params:
        linearSVM = LinearSVC(C=C, class_weight='balanced')
        linearSVM = linearSVM.fit(prepared_train_data, prepared_train_labels)
        # validation with recovering the number of frames of each video
        """score = validate_estimator_on_dict(estimator=linearSVM, val_data=val_data, sample_rates=sample_rates,
                                           original_labels=val_labels,
                                           window_size=window_size, window_step=window_step,
                                           plot_and_save_conf_matrix=True)
        print('C:', C, 'score:', score)"""
        #save_recovered_predictions_to_dir(path_to_save=path_to_save_predictions, estimator=linearSVM, val_data=val_data,
        #                                  sample_rates=sample_rates, original_labels=val_labels,
        #                                  window_size=window_size, window_step=window_step)
        if generate_test_predictions:
            del train_data
            del val_data
            del prepared_train_data
            #del val_data
            gc.collect()
            # load val data
            test_data = np.load(load_path_test_data, allow_pickle=True)
            test_labels = pd.read_csv(load_path_test_labels)
            test_labels['name_folder'] = test_labels['name_folder'].astype('str')
            test_data, test_labels = transform_data_and_labels_to_dict(test_data, test_labels, sample_rates)
            test_data, _ = prepare_data_and_labels_for_svm(
                data=test_data, labels=test_labels.copy(), window_size=window_size, window_step=window_step,
                normalization=True, normalization_types=normalization_types, return_normalizers=False,
                normalizers=train_normalizers, class_to_delete=None)
            # save recovered test predictions
            save_recovered_predictions_to_dir(path_to_save=path_to_save_test_predictions, estimator=linearSVM,
                                              val_data=test_data,
                                              sample_rates=sample_rates, original_labels=test_labels,
                                              window_size=window_size, window_step=window_step)


if __name__ == '__main__':
    main([0.1], 4, 2, normalization_types=('z','l2'), generate_test_predictions=True,
         path_to_save_test_predictions='Linear_SVM_4_2_z_l2')
    main([0.01], 4, 2, normalization_types=('z','power','l2'), generate_test_predictions=True,
         path_to_save_test_predictions='Linear_SVM_4_2_z_power_l2')
    print("################################")
    """main(4,2, normalization_types=('z','power','l2'))
    print("################################")
    main(2,1, normalization_types=('z','l2'))
    print("################################")
    main(2,1, normalization_types=('z','power','l2'))
    print("################################")"""

    """path_to_save_features='saved_features'
    load_path_train_data = 'D:\\Downloads\\aff_wild2_train_emo_with_loss.pickle'
    load_path_train_labels = 'D:\\Downloads\\df_affwild2_train_emo.csv'
    load_path_val_data = 'D:\\Downloads\\aff_wild2_val_emo_with_loss.pickle'
    load_path_val_labels = 'D:\\Downloads\\df_affwild2_val_emo.csv'
    load_path_sample_rates = 'D:\\Downloads\\videos_frame_rate.txt'
    window_size = 4.
    window_step = 2.
    normalization_types = ("",)
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
        normalization=True, normalization_types=('z','l2'), return_normalizers=True,
        normalizers=None, class_to_delete=-1)
    # clear RAM
    gc.collect()
    # save features
    if not os.path.exists(path_to_save_features):
        os.mkdir(path_to_save_features)
    if not os.path.exists(os.path.join(path_to_save_features, 'train')):
        os.mkdir(os.path.join(path_to_save_features, 'train'))
    save_features_to_file(os.path.join(path_to_save_features, 'train'), prepared_train_data, prepared_train_labels)
    del prepared_train_labels
    del prepared_train_data
    del train_data
    del train_labels
    # all the same operations for validation data
    # load val data
    val_data = np.load(load_path_val_data, allow_pickle=True)
    val_labels = pd.read_csv(load_path_val_labels)
    val_labels['name_folder'] = val_labels['name_folder'].astype('str')
    val_data, val_labels = transform_data_and_labels_to_dict(val_data, val_labels, sample_rates)
    val_data, _ = prepare_data_and_labels_for_svm(
        data=val_data, labels=val_labels.copy(), window_size=window_size, window_step=window_step,
        normalization=True, normalization_types=None, return_normalizers=False,
        normalizers=train_normalizers, class_to_delete=None)

    # save features
    if not os.path.exists(path_to_save_features):
        os.mkdir(path_to_save_features)
    if not os.path.exists(os.path.join(path_to_save_features, 'val')):
        os.mkdir(os.path.join(path_to_save_features, 'val'))
    save_features_to_file(os.path.join(path_to_save_features, 'val'), val_data, _)

    prepared_val_data, prepared_val_labels = concatenate_all_data_and_labels(val_data,
                                                                                 _)
    visualize_features_according_class(prepared_val_data, prepared_val_labels)"""
