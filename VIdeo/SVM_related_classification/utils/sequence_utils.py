# TODO: write description
from typing import List, Tuple
import numpy as np
from numpy.polynomial import polynomial

def extract_polynomial_coefficients_from_2d_window(window:np.ndarray, order:int):
    coefficients=[]
    x=np.linspace(0,1,window.shape[0])
    for feature_idx in range(window.shape[1]):
        y=window[:,feature_idx]
        coeffs=polynomial.polyfit(x, y, deg=order)[order]
        coefficients.append(coeffs)
    coefficients=np.array(coefficients)
    return coefficients

def extract_statistics_from_2d_window(window:np.ndarray,
                                      statistic_types:Tuple[str,...]=('mean', 'std', 'poly_1d','poly_2d')) -> np.ndarray:
    """Extracts from 2d window functionals for each feature across instances (rows).
    Currently supported functionals: ('mean', 'std')

    :param window: np.ndarray
                data in 2-d format (n_samples, n_features)
    :param statistic_types: Tuple[str,...]
                required functionals to extract.
    :return: np.ndarray
                extracted functionals with shape (n_samples, n_functionals)
    """
    supported_statistics=('mean', 'std', 'poly_1d','poly_2d')
    # check if statistic_types contains only supported functional types
    if not set(statistic_types).issubset(supported_statistics):
        raise AttributeError('Currently supported statistics:%s. Got %s.'%(supported_statistics, statistic_types))
    if len(window.shape)!=2:
        raise AttributeError('The window should be 2-dimensional. Got %i dimensions.'%(len(window.shape)))

    result_statistics=[]
    for statistic_type in statistic_types:
        if statistic_type=='mean':
            statistics=window.mean(axis=0)
        elif statistic_type=='std':
            statistics=window.std(axis=0)
        elif statistic_type=='poly_1d':
            statistics=extract_polynomial_coefficients_from_2d_window(window, order=1)
        elif statistic_type=='poly_2d':
            statistics=extract_polynomial_coefficients_from_2d_window(window, order=2)
        result_statistics.append(statistics)
    result_statistics=np.concatenate(result_statistics, axis=0)
    result_statistics=result_statistics[..., np.newaxis]
    return result_statistics


def cut_data_on_chunks(data:np.ndarray, chunk_length:int, window_step:int) -> List[np.ndarray]:
    """Cuts data on chunks according to supplied chunk_length and windows_step.
        Example:
        data=|123456789|, chunk_length=4, window_step=3
        result= [1234, 4567, 6789] -> last element (6789)

    :param data: ndarray
                sequence to cut
    :param chunk_length: int
                length of window/chunk. It is calculated before function as seconds*sample_rate
    :param window_step: int
                length of shift of the window/chunk.
    :return: list of np.ndarrays
                cut on windows data
    """
    if data.shape[0]<chunk_length:
        raise AttributeError("data length should be >= chunk length. Got data length:%i, chunk length:%i"%(data.shape[0], chunk_length))
    if data.shape[0]<window_step:
        raise AttributeError("data length should be >= window_step. Got data length:%i, window_step:%i"%(data.shape[0], window_step))

    cut_data=[]
    num_chunks=int(np.ceil((data.shape[0]-chunk_length)/window_step+1))

    for chunk_idx in range(num_chunks-1):
        start=chunk_idx*window_step
        end=chunk_idx*window_step+chunk_length
        chunk=data[start:end]
        cut_data.append(chunk)

    last_chunk=data[-chunk_length:]
    cut_data.append(last_chunk)
    return cut_data


if __name__ == '__main__':
    pass