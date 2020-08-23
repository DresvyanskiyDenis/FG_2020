import os

import pandas as pd
import numpy as np
from scipy.io import wavfile

# TODO: реализовать функцию загрузки данных и лэйюлов в классе так, чтобы можно было падавать любую функцию в них.
from sklearn.metrics import accuracy_score, f1_score

def generate_weights(amount_class_array):
    result_weights=amount_class_array/np.sum(amount_class_array)
    result_weights=1./result_weights
    result_weights=result_weights/np.sum(result_weights)
    return result_weights

def load_labels(path_to_labels):
    f = open(path_to_labels, 'r')
    original_sample_rate = int(f.readline().split(':')[-1])
    f.close()
    labels=pd.read_csv(path_to_labels, skiprows=1,header=None)
    return labels.values.reshape((-1,)), original_sample_rate

def load_data(path_to_datafile, filetype):
    if filetype == 'wav':
        sample_rate, data = wavfile.read(path_to_datafile)
        return data, sample_rate
    elif filetype =='csv':
        data=pd.read_csv(path_to_datafile ,header=None)
        return data.values, None

def how_many_windows_do_i_need(length_sequence, window_size, step):
    """This function calculates how many windows do you need
        with corresponding length of sequence, window_size and
        window_step
        for example, if your sequence length=10, window_size=4 and
        window_step=2 then:
        |_ _ _ _| _ _ _ _ _ _
        _ _ |_ _ _ _| _ _ _ _
        _ _ _ _ |_ _ _ _| _ _
        _ _ _ _ _ _ |_ _ _ _|
        ==> you need 4 windows with this parameters

    :param length_sequence: int, the length of sequence
    :param window_size: int, the length of window
    :param step: int
    :return: int, number of windows needed for this sequence
    """
    start_idx=0
    counter=0
    while True:
        if start_idx+window_size>=length_sequence:
            break
        start_idx+=step
        counter+=1
    if start_idx!=length_sequence:
        counter+=1
    return counter

class Database():

    def __init__(self, path_to_data, path_to_labels, data_filetype='wav', data_postfix=''):

        self.path_to_data=path_to_data
        self.path_to_labels=path_to_labels
        self.data_frame_rate=None
        self.labels_frame_rate=None
        self.data_instances=[]
        self.data_filetype=data_filetype
        self.data_postfix=data_postfix

    def load_all_data_and_labels(self):
        """This function loads data and labels from folder self.path_to_data and file with path path_to_labels
           For computational efficiency the loading of labels is made as a separate function load_labels_get_dict()
           Every file is represented as instance of class Database_instance(). The data loading realized by Database_instance() class.
           Since all files have the same frame rates (as well as labels), data_frame_rate and labels_frame_rate will set
           to the same value taken from first element of list data_instances

        :return:None
        """
        # Since all labels are represented by only one file, for computational effeciency firstly we load all labels
        # and then give them to different loaded audiofiles
        list_labels_filenames=os.listdir(self.path_to_labels)
        for labels_filename in list_labels_filenames:
            instance = Database_instance()
            instance.load_data(self.path_to_data + labels_filename.split('_left')[0].split('_right')[0].split('.')[0]+self.data_postfix+'.'+self.data_filetype, self.data_filetype)
            instance.labels, instance.labels_frame_rate=load_labels(self.path_to_labels+labels_filename)
            instance.generate_timesteps_for_labels()
            self.data_instances.append(instance)
        self.data_frame_rate=self.data_instances[0].data_frame_rate
        self.labels_frame_rate = self.data_instances[0].labels_frame_rate

    def cut_all_instances(self, window_size, window_step):
        """This function is cutting all instances of database (elements of list, which is Database_instance())
        It exploits included in Database_instance() class function for cutting.

        :param window_size: float, size of window in seconds
        :param window_step: float, step of window in seconds
        :return: None
        """
        for i in range(len(self.data_instances)):
            self.data_instances[i].cut_data_and_labels_on_windows(window_size, window_step)

    def get_all_concatenated_cutted_data_and_labels(self):
        """This function concatenates cutted data and labels of all elements of list self.data_instances
           Every element of list is Database_instance() class, which contains field cutted_data and cutted_labels

        :return: 2D ndarray, shape=(num_instances_in_list*num_windows_per_instance, data_window_size),
                    concatenated cutted_data of every element of list self.data_instances
                 2D ndarray, shape=(num_instances_in_list*num_windows_per_instance, labels_window_size),
                    concatenated cutted_labels of every element of list self.data_instances
        """
        tmp_data=[]
        tmp_labels=[]
        for i in range(len(self.data_instances)):
            tmp_data.append(self.data_instances[i].cutted_data)
            tmp_labels.append(self.data_instances[i].cutted_labels)
        result_data=np.vstack(tmp_data)
        result_labels = np.vstack(tmp_labels)
        return result_data, result_labels

    def get_all_concatenated_data_and_labels(self):
        """This function concatenates data and labels of all elements of list self.data_instances
           Every element of list is Database_instance() class, which contains field data and labels

        :return: 2D ndarray, shape=(num_instances_in_list*num_per_instance, data_window_size),
                    concatenated data of every element of list self.data_instances
                 2D ndarray, shape=(num_instances_in_list*num_per_instance, labels_window_size),
                    concatenated labels of every element of list self.data_instances
        """
        tmp_data=[]
        tmp_labels=[]
        for i in range(len(self.data_instances)):
            tmp_data.append(self.data_instances[i].data)
            tmp_labels.append(self.data_instances[i].labels.reshape((-1,1)))
        result_data=np.vstack(tmp_data)
        result_labels = np.vstack(tmp_labels).reshape((-1))
        return result_data, result_labels

    def get_all_concatenated_cutted_labels_timesteps(self):
        tmp_timesteps=[]
        for i in range(len(self.data_instances)):
            tmp_timesteps.append(self.data_instances[i].cutted_labels_timesteps)
        result_timesteps=np.vstack(tmp_timesteps)
        return result_timesteps

    def reduce_labels_frame_rate(self, needed_frame_rate):
        """This function reduce labels frame rate to needed frame rate by taking every (second, thirs and so on) elements from
           based on calculated ratio.
           ratio calculates between current frame rate and needed frame rate

        :param needed_frame_rate: int, needed frame rate of labels per one second (e.g. 25 labels per second)
        :return:None
        """
        ratio=int(self.labels_frame_rate/needed_frame_rate)
        self.labels_frame_rate=needed_frame_rate
        for i in range(len(self.data_instances)):
            self.data_instances[i].labels=self.data_instances[i].labels[::ratio]
            self.data_instances[i].labels_frame_rate=needed_frame_rate


    def shuffle_and_separate_cutted_data_on_train_and_val_sets(self, percent_of_validation):
        """This function shuffle and then separate cutted data and labels by given percent_of_validation
           It exploits class function get_all_concatenated_cutted_data_and_labels() to get cutted data and labels from
           each database_instance and then concatenate it
           Then resulted arrays of get_all_concatenated_cutted_data_and_labels() function will be
           shuffled and then separated on train and validation parts

        :param percent_of_validation: float, percent of validation part in all data
        :return: 2D ndarray, shape=(num_instances_in_list*num_windows_per_instance*(100-percent_of_validation)/100, data_window_size),
                    train data - concatenated cutted_data of every element of list self.data_instances
                 2D ndarray, shape=(num_instances_in_list*num_windows_per_instance*(100-percent_of_validation)/100, labels_window_size),
                    train labels - concatenated cutted_labels of every element of list self.data_instances
                 2D ndarray, shape=(num_instances_in_list*num_windows_per_instance*percent_of_validation/100, data_window_size),
                    validation data - concatenated cutted_data of every element of list self.data_instances
                 2D ndarray, shape=(num_instances_in_list*num_windows_per_instance*percent_of_validation/100, labels_window_size),
                    validation labels - concatenated cutted_data of every element of list self.data_instances
        """
        concatenated_data, concatenated_labels=self.get_all_concatenated_cutted_data_and_labels()
        permutation=np.random.permutation(concatenated_data.shape[0])
        concatenated_data, concatenated_labels= concatenated_data[permutation], concatenated_labels[permutation]
        sep_idx=int(concatenated_data.shape[0]*(100-percent_of_validation)/100.)
        train_part_data, train_part_labels= concatenated_data[:sep_idx], concatenated_labels[:sep_idx]
        val_part_data, val_part_labels= concatenated_data[sep_idx:], concatenated_labels[sep_idx:]
        return train_part_data, train_part_labels, val_part_data, val_part_labels

    def prepare_data_for_training(self, window_size, window_step):
        # aligning labels
        for instance in self.data_instances:
            instance.align_number_of_labels_and_data()
        # delete all -1 labels
        for instance in self.data_instances:
            instance.data=instance.generate_array_without_class(instance.data, -1)
            instance.labels_timesteps = instance.generate_array_without_class(instance.labels_timesteps, -1)
            instance.labels = instance.generate_array_without_class(instance.labels, -1)

        # check if some file have 0 labels (this could be, if all labels were -1. You can face it in FG_2020 competition)
        tmp_list=[]
        for instance in self.data_instances:
            if instance.labels.shape[0]!=0:
                tmp_list.append(instance)
        self.data_instances=tmp_list

        # cutting
        self.cut_all_instances(window_size, window_step)






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

    def load_data(self, path_to_data, data_filetype):
        """ This function load data and corresponding frame rate from wav type file

        :param path_to_data: String
        :return: None
        """
        self.data, self.data_frame_rate=load_data(path_to_data, data_filetype)
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
        if self.data_frame_rate==None:
            self.data_frame_rate = self.labels_frame_rate
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


    def load_labels(self, path_to_labels):
        """This function loads labels for certain, concrete audiofile
        It exploits load_labels_get_dict() function, which loads and parses all labels from one label-file
        Then we just take from obtained dictionary labels by needed audio filename.
        Current solution is computational unefficient, but it is used very rarely

        :param path_to_labels:String
        :return:None
        """
        self.labels, self.labels_frame_rate=load_labels(path_to_labels)


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
        aligned_labels = np.zeros(shape=(self.data.shape[0]), dtype='int32')
        if self.data.shape[0] <= self.labels.shape[0]:
            aligned_labels[:] = self.labels[:self.data.shape[0]]
        else:
            aligned_labels[:self.labels.shape[0]] = self.labels[:]
            value_to_fill = self.labels[-1]
            aligned_labels[self.labels.shape[0]:] = value_to_fill
        self.labels=aligned_labels
        self.generate_timesteps_for_labels()

    def generate_array_without_class(self, arr, class_num):
        indexes=self.labels!=class_num
        return arr[indexes]







class Metric_calculator():
    """This class is created to calculate metrics.
       Moreover, it can average cutted predictions with the help of their  cutted_labels_timesteps.
       cutted_labels_timesteps represents timestep of each cutted prediction value. Cutted predictions comes from
       model, which can predict values from data only partual, with defined window size
       e.g. we have
        cutted_prediction=np.array([
            [1, 2, 3, 4, 5],
            [6, 5 ,43, 2, 5],
            [2, 65, 1, 4, 6],
            [12, 5, 6, 34, 23]
        ])
        cutted_labels_timesteps=np.array([
            [0,  0.2, 0.4, 0.6, 0.8],
            [0.2, 0.4, 0.6, 0.8, 1],
            [0.4, 0.6, 0.8, 1, 1.2],
            [0.6, 0.8, 1, 1.2, 1.4],
        ])

    it takes, for example all predictions with timestep 0.2 and then average it -> (2+6)/2=4
    the result array will:
    self.predictions=[ 1.0, 4.0, 3.333, 31.0, 3.25, 5.0, 20.0, 23.0]
    timesteps=       [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]


    """

    def __init__(self, cutted_predictions, cutted_labels_timesteps, ground_truth):
        self.ground_truth=ground_truth
        self.predictions=None
        self.cutted_predictions=cutted_predictions
        self.cutted_labels_timesteps=cutted_labels_timesteps

    def average_cutted_predictions_by_timestep(self, mode='regression'):
        """This function averages cutted predictions. For more info see description of class
        :return: None
        """
        if mode=='regression':
            cutted_predictions_flatten=self.cutted_predictions.flatten()[..., np.newaxis]
            cutted_labels_timesteps_flatten=self.cutted_labels_timesteps.flatten()[..., np.newaxis]
            dataframe_for_avg=pd.DataFrame(columns=['prediction','timestep'], data=np.concatenate((cutted_predictions_flatten, cutted_labels_timesteps_flatten), axis=1))
            dataframe_for_avg=dataframe_for_avg.groupby(by=['timestep']).mean()
            self.predictions=dataframe_for_avg['prediction'].values
        elif mode=='categorical_probabilities':
            cutted_predictions_flatten=self.cutted_predictions.reshape((-1, self.cutted_predictions.shape[-1]))
            cutted_labels_timesteps_flatten=self.cutted_labels_timesteps.reshape((-1,1))
            dataframe_for_avg=pd.DataFrame(data=np.concatenate((cutted_labels_timesteps_flatten, cutted_predictions_flatten), axis=1))
            dataframe_for_avg=dataframe_for_avg.rename(columns={0:'timestep'})
            dataframe_for_avg = dataframe_for_avg.groupby(by=['timestep']).mean()
            predictions_probabilities=dataframe_for_avg.iloc[:].values
            predictions_probabilities=np.argmax(predictions_probabilities, axis=-1)
            self.predictions=predictions_probabilities


    def calculate_FG_2020_categorical_score_across_all_instances(self, instances):
        # TODO: peredelat na bolee logichniy lad. Eto poka chto bistroo reshenie
        ground_truth_all=np.zeros((0,))
        predictions_all=np.zeros((0,))
        for instance in instances:
            ground_truth_all=np.concatenate((ground_truth_all, instance.labels))
            predictions_all = np.concatenate((predictions_all, instance.predictions))
        return f1_score(ground_truth_all, predictions_all, average='macro')


    def calculate_accuracy(self):
        return accuracy_score(self.ground_truth, self.predictions)

    def calculate_f1_score(self, mode='macro'):
        return f1_score(self.ground_truth, self.predictions, average=mode)

    def calculate_FG_2020_categorical_score(self, f1_score_mode='macro'):
        return 0.67*self.calculate_f1_score(f1_score_mode)+0.33*self.calculate_accuracy()