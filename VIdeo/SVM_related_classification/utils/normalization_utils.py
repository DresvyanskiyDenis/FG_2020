# TODO: write description



from typing import Optional, Union, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler, normalize, PowerTransformer


def z_normalization(data:np.ndarray, return_scaler:bool=False,
                    scaler:Optional[object]=None)->Union[np.ndarray, Tuple[np.ndarray, object]]:
    # TODO: write description
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
    # TODO: write description
    # check if data is in appropriate format
    if len(data.shape) > 2:
        raise AttributeError('The supplied data should be 1- or 2-dimensional. Got %i.' % (len(data.shape)))
    # if data is 1-dimensional, it should be converted into 2-dimensional by adding additional dimension
    if len(data.shape) == 1:
        data = data[..., np.newaxis]
    # normalize data. axis=1 means that each sample will be independently normalized.
    data=normalize(data, axis=1)
    return data

def power_normalization(data:np.ndarray, return_scaler:bool=False,
                        scaler:Optional[object]=None)->Union[np.ndarray, Tuple[np.ndarray, object]]:
    # TODO: write description
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