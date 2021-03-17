# TODO: write description



from typing import Optional, Union, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler, normalize, PowerTransformer


def z_normalization(data:np.ndarray, return_scaler:bool=False,
                    scaler:Optional[object]=None)->Union[np.ndarray, Tuple[np.ndarray, object]]:
    """Normalizes data with provided normalizer, or generates new one and fit it.
    Uses sklearn.preprocessing.StandardScaler() object.

    :param data: np.ndarray
                data in 2d-format (n_samples, n_features) or 1d-format (n_samples,)
    :param return_scaler: bool
                Should function return fitted scaler or not.
    :param scaler: object (sklearn.preprocessing.StandardScaler)
                If not None, the provided scaler will be used as normalizer.
    :return: np.ndarray or (np.ndarray, scaler)
                If return_scaler==False, returns normalized data
                else returns normalized data and fitted scaler
    """
    # check if data is in appropriate format
    if len(data.shape)>2:
        raise AttributeError('The supplied data should be 1- or 2-dimensional. Got %i.'%(len(data.shape)))
    # if data is 1-dimensional, it should be converted into 2-dimensional by adding additional dimension
    if len(data.shape)==1:
        data=data[..., np.newaxis]

    # if no scaler supplied, create scaler and fit it
    if scaler is None:
        scaler=StandardScaler()
        scaler.fit(data)
    # transform data
    data=scaler.transform(data)
    # return scaler if need
    if return_scaler:
        return data, scaler
    return data

def l2_normalization(data:np.ndarray)->np.ndarray:
    """Normalizes data with l2 normalization. Each instance (vector) will be normalized independently.
        Uses sklearn.preprocessing.normalize() function.

    :param data: np.ndarray
                data in 2d-format (n_samples, n_features) or 1d-format (n_samples,)
    :return: np.ndarray
                normalized data
    """
    # check if data is in appropriate format
    if len(data.shape) > 2:
        raise AttributeError('The supplied data should be 1- or 2-dimensional. Got %i.' % (len(data.shape)))
    # if data is 1-dimensional, it should be converted into 2-dimensional by adding additional dimension
    if len(data.shape) == 1:
        data = data[..., np.newaxis]
    # normalize data. axis=1 means that each instance (row) will be independently normalized.
    data=normalize(data, axis=1)
    return data


def power_normalization(data:np.ndarray, return_scaler:bool=False,
                        scaler:Optional[object]=None)->Union[np.ndarray, Tuple[np.ndarray, object]]:
    """Normalizes provided data via power normalization.
    More: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    Uses sklearn.preprocessing.PowerTransformer() object.
    :param data: np.ndarray
                data in 2d-format (n_samples, n_features) or 1d-format (n_samples,)
    :param return_scaler: bool
                Should function return fitted scaler or not.
    :param scaler: object (sklearn.preprocessing.StandardScaler)
                If not None, the provided scaler will be used as normalizer.
    :return: np.ndarray or (np.ndarray, scaler)
                If return_scaler==False, returns normalized data
                else returns normalized data and fitted scaler
    """
    # check if data is in appropriate format
    if len(data.shape) > 2:
        raise AttributeError('The supplied data should be 1- or 2-dimensional. Got %i.' % (len(data.shape)))
    # if data is 1-dimensional, it should be converted into 2-dimensional by adding additional dimension
    if len(data.shape) == 1:
        data = data[..., np.newaxis]

    # if no scaler supplied, create scaler and fit it
    if scaler is None:
        scaler = PowerTransformer()
        scaler.fit(data)
    # transform data
    data = scaler.transform(data)
    # return scaler if need
    if return_scaler:
        return data, scaler
    return data