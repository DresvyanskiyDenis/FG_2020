import math

from Audio.Preprocessing.labels_utils import extend_sample_rate_of_labels
from Audio.Training.utils import how_many_windows_do_i_need
import numpy as np

class Database_instance():
    """This class represents one instance of database,
       including data and labels"""
    def __init__(self):
        self.filename=None
        self.data_window_size = None
        self.data_window_step = None
        self.labels_window_size = None
        self.labels_window_step = None
        self.data = None
        self.data_frame_rate=None
        self.cutted_data = None
        self.labels = None
        self.labels_frame_rate=None
        self.cutted_labels = None
        self.labels_timesteps= None
        self.cutted_labels_timesteps=None
        self.cutted_predictions=None
        self.predictions=None
        self.loading_data_function=None

    def load_data(self, path_to_data):
        """ This function load data and corresponding frame rate from wav type file

        :param path_to_data: String
        :return: None
        """
        self.data, self.data_frame_rate=self.loading_data_function(path_to_data)
        self.filename=path_to_data.split('\\')[-1].split('/')[-1].split('.')[0]


    def pad_the_sequence(self, sequence, window_size,  mode, padding_value=0):
        """This fucntion pad sequence with corresponding padding_value to the given shape of window_size
        For example, if we have sequence with shape 4 and window_size=6, then
        it just concatenates 2 specified values like
        to the right, if padding_mode=='right'
            last_step   -> _ _ _ _ v v  where v is value (by default equals 0)
        to the left, if padding_mode=='left'
            last_step   -> v v _ _ _ _  where v is value (by default equals 0)
        to the center, if padding_mode=='center'
            last_step   -> v _ _ _ _ v  where v is value (by default equals 0)

        :param sequence: ndarray
        :param window_size: int
        :param mode: string, can be 'right', 'left' or 'center'
        :param padding_value: float
        :return: ndarray, padded to given window_size sequence
        """
        result=np.ones(shape=(window_size))*padding_value
        if mode=='left':
            result[(window_size-sequence.shape[0]):]=sequence
        elif mode=='right':
            result[:sequence.shape[0]]=sequence
        elif mode=='center':
            start=(window_size-sequence.shape[0])//2
            end=start+sequence.shape[0]
            result[start:end]=sequence
        else:
            raise AttributeError('mode can be either left, right or center')
        return result

    def cut_sequence_on_windows(self, sequence, window_size, window_step):
        """This function cuts given sequence on windows with corresponding window_size and window_step
        for example, if we have sequence [1 2 3 4 5 6 7 8], window_size=4, window_step=3 then
        1st step: |1 2 3 4| 5 6 7 8
                  ......
        2nd step: 1 2 3 |4 5 6 7| 8
                        ..
        3rd step: 1 2 3 4 |5 6 7 8|

        Here, in the last step, if it is not enough space for window, we just take window, end of which is last element
        In given example for it we just shift window on one element
        In future version maybe padding will be added
        :param sequence: ndarray
        :param window_size: int, size of window
        :param window_step: int, step of window
        :return: 2D ndarray, shape=(num_windows, window_size)
        """
        num_windows = how_many_windows_do_i_need(sequence.shape[0], window_size, window_step)
        if len(sequence.shape)>1:
            cutted_data=np.zeros(shape=(num_windows, window_size)+sequence.shape[1:])
        else:
            cutted_data = np.zeros(shape=(num_windows, window_size))

        # if sequence has length less than whole window
        if sequence.shape[0]<window_size:
            cutted_data[0, :sequence.shape[0]]=sequence
            return cutted_data


        start_idx=0
        # start of cutting
        for idx_window in range(num_windows-1):
            end_idx=start_idx+window_size
            cutted_data[idx_window]=sequence[start_idx:end_idx]
            start_idx+=window_step
        # last window
        end_idx=sequence.shape[0]
        start_idx=end_idx-window_size
        cutted_data[num_windows-1]=sequence[start_idx:end_idx]
        return cutted_data

    def cut_data_and_labels_on_windows(self, window_size, window_step):
        """This function exploits function cut_sequence_on_windows() for cutting data and labels
           with corresponding window_size and window_step
           Window_size and window_step are calculating independently corresponding data and labels frame rate

        :param window_size: float, size of window in seconds
        :param window_step: float, step of window in seconds
        :return: 2D ndarray, shape=(num_windows, data_window_size), cutted data
                 2D ndarray, shape=(num_windows, labels_window_size), cutted labels
        """
        # calculate params for cutting (size of window and step in index)
        self.data_window_size=int(window_size*self.data_frame_rate)
        self.data_window_step=int(window_step*self.data_frame_rate)
        self.labels_window_size=int(window_size*self.labels_frame_rate)
        self.labels_window_step=int(window_step*self.labels_frame_rate)

        self.cutted_data=self.cut_sequence_on_windows(self.data, self.data_window_size, self.data_window_step)
        self.cutted_labels=self.cut_sequence_on_windows(self.labels, self.labels_window_size, self.labels_window_step)
        self.cutted_labels_timesteps=self.cut_sequence_on_windows(self.labels_timesteps, self.labels_window_size, self.labels_window_step)
        self.cutted_data = self.cutted_data.astype('float32')
        self.cutted_labels = self.cutted_labels.astype('int32')
        self.cutted_labels_timesteps= self.cutted_labels_timesteps.astype('float32')
        return self.cutted_data, self.cutted_labels, self.cutted_labels_timesteps


    def generate_timesteps_for_labels(self):
        """This function generates timesteps for labels with corresponding labels_frame_rate
           After executing it will be saved in field self.labels_timesteps
        :return: None
        """
        label_timestep_in_sec=1./self.labels_frame_rate
        timesteps=np.array([i for i in range(self.labels.shape[0])], dtype='float32')
        timesteps=timesteps*label_timestep_in_sec
        self.labels_timesteps=timesteps

    def align_number_of_labels_and_data(self):
        if self.data_frame_rate==self.labels_frame_rate:
            new_shape_labels=self.data.shape[0]
        else:
            new_shape_labels=int(math.ceil(self.data.shape[0]/self.data_frame_rate*self.labels_frame_rate))
        aligned_labels = np.zeros(shape=(new_shape_labels), dtype='int32')
        if new_shape_labels <= self.labels.shape[0]:
            aligned_labels[:] = self.labels[:new_shape_labels]
        else:
            aligned_labels[:self.labels.shape[0]] = self.labels[:]
            value_to_fill = self.labels[-1]
            aligned_labels[self.labels.shape[0]:] = value_to_fill
        self.labels=aligned_labels
        self.generate_timesteps_for_labels()

    def generate_array_without_class(self, arr,arr_frame_rate, class_num):
        # TODO: upgrade this function in more efficient and explicit way
        indexes=self.labels!=class_num
        if arr_frame_rate!=self.labels_frame_rate:
            indexes=extend_sample_rate_of_labels(indexes, self.labels_frame_rate, arr_frame_rate).astype('bool')
            indexes=indexes[:arr.shape[0]]
        return arr[indexes]

