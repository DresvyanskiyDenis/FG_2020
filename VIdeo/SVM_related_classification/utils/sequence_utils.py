import numpy as np





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