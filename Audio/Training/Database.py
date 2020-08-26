import numpy as np
import os

from sklearn.preprocessing import StandardScaler

from Audio.Training.Database_instance import Database_instance


class Database():

    def __init__(self, path_to_data, path_to_labels, data_filetype='wav', data_postfix=''):

        self.path_to_data=path_to_data
        self.path_to_labels=path_to_labels
        self.data_frame_rate=None
        self.labels_frame_rate=None
        self.data_instances=[]
        self.data_filetype=data_filetype
        self.data_postfix=data_postfix

    def load_all_data_and_labels(self, loading_data_function, loading_labels_function):
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
            instance.loading_data_function=loading_data_function
            instance.load_data(self.path_to_data + labels_filename.split('_left')[0].split('_right')[0].split('.')[0]+self.data_postfix+'.'+self.data_filetype)
            instance.labels, instance.labels_frame_rate=loading_labels_function(self.path_to_labels+labels_filename)
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

    def normalize_data_within_database(self, scaler=None, return_scaler=False):
        data, _ = self.get_all_concatenated_data_and_labels()
        if scaler==None:
            scaler=StandardScaler()
            scaler=scaler.fit(data)
        for instance in self.data_instances:
            instance.data=scaler.transform(instance.data)
        if return_scaler==True:
            return scaler

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

    def prepare_data_for_training(self, window_size, window_step, delete_value=-1, need_scaling=False, scaler=None, return_scaler=False):
        # aligning labels
        for instance in self.data_instances:
            instance.align_number_of_labels_and_data()
        # delete all -1 labels
        if delete_value!=None:
            for instance in self.data_instances:
                instance.data=instance.generate_array_without_class(instance.data,instance.data_frame_rate, delete_value)
                instance.labels_timesteps = instance.generate_array_without_class(instance.labels_timesteps,instance.labels_frame_rate, delete_value)
                instance.labels = instance.generate_array_without_class(instance.labels,instance.labels_frame_rate, delete_value)

        # check if some file have 0 labels (this could be, if all labels were -1. You can face it in FG_2020 competition)
        tmp_list=[]
        for instance in self.data_instances:
            if instance.labels.shape[0]!=0:
                tmp_list.append(instance)
        self.data_instances=tmp_list

        # scaling
        if need_scaling==True:
            scaler=self.normalize_data_within_database(scaler=scaler, return_scaler=return_scaler)
        # cutting
        self.cut_all_instances(window_size, window_step)
        # return scaler if need
        if return_scaler==True:
            return scaler