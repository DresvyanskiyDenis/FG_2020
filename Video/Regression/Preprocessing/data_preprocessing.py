import math
from shutil import copyfile

import PIL
import cv2
import pandas as pd
import numpy as np
import os

from Audio.utils import utils

def construct_video_filename_from_label(path_to_video, label_filename):
    """This function generate video filename from label filename
       It is inner function for processing specific data from AffWild2 challenge
       It is needed, because video can be in either mp4 or avi format

    :param path_to_video: string, path to directory with videos
    :param label_filename: string, filename of labels
    :return: string, video filename (e. g. 405.mp4)
    """
    video_filename = label_filename.split('_left')[0].split('_right')[0].split('.')[0]
    if os.path.exists(path_to_video + video_filename + '.mp4'):
        video_filename += '.mp4'
    if os.path.exists(path_to_video + video_filename + '.avi'):
        video_filename += '.avi'
    return video_filename

def get_video_frame_rate(path_to_video):
    """The function reads params of video to get video frame rate

    :param path_to_video: string, path to certain video
    :return: int, video frame rate
    """
    cap = cv2.VideoCapture(path_to_video)
    video_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    return video_frame_rate

def add_missing_frames(path_to_frames, number_must_be):
    indexes=np.array([i+1 for i in range(number_must_be)])
    if os.path.exists(path_to_frames+'.DS_Store'):
        os.remove(path_to_frames+'.DS_Store')
    frames=os.listdir(path_to_frames)
    filename_to_add_missing_frame=frames[0]
    for idx in indexes:
        filename=str(idx).zfill(5)+'.jpg'
        if not os.path.exists(path_to_frames+filename):
            copyfile(path_to_frames+filename_to_add_missing_frame, path_to_frames+filename)
        else:
            filename_to_add_missing_frame=filename

def add_missing_frames_in_all_folders(paths_to_folders, path_to_labels):
    labels_filenames=os.listdir(path_to_labels)
    counter=0
    for label_filename in labels_filenames:
        lbs=pd.read_csv(path_to_labels+label_filename, header=None)
        number_must_be=lbs.shape[0]
        # get video frames
        filename_video=construct_video_filename_from_label('D:\\Databases\\AffWild2\\Videos\\', label_filename)
        cap = cv2.VideoCapture('D:\\Databases\\AffWild2\\Videos\\'+filename_video)
        video_frame_rate = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        path_to_folder_with_frames=paths_to_folders+label_filename.split('.')[0]+'\\'
        add_missing_frames(path_to_frames=path_to_folder_with_frames,
                           number_must_be=number_must_be)
        counter+=1
        print('processed:%i, remaining:%i'%(counter, len(labels_filenames)-counter))


def cut_sequence_on_windows(sequence, window_size, window_step):
    num_windows = utils.how_many_windows_do_i_need(sequence.shape[0], window_size, window_step)
    cutted_data_indexes=np.zeros(shape=(num_windows, 2), dtype='int32')

    # if sequence has length less than whole window
    if sequence.shape[0]<window_size:
        cutted_data_indexes[0, 0]=0
        cutted_data_indexes[0, 1] = window_size
        return cutted_data_indexes


    start_idx=0
    # start of cutting
    for idx_window in range(num_windows-1):
        end_idx=start_idx+window_size
        cutted_data_indexes[idx_window,0]=start_idx
        cutted_data_indexes[idx_window, 1] = end_idx
        start_idx+=window_step
    # last window
    end_idx=sequence.shape[0]
    start_idx=end_idx-window_size
    cutted_data_indexes[-1, 0] = start_idx
    cutted_data_indexes[-1, 1] = end_idx
    return cutted_data_indexes


def thin_array_indexes(array, original_sample_rate, needed_sample_rate):
    ratio=needed_sample_rate/original_sample_rate
    new_size=int(math.ceil(array.shape[0]*ratio))
    indexes=np.zeros((new_size,), dtype='int32')
    indexes[0]=0
    idx_array_indexes=1
    counter=0
    for i in range(array.shape[0]):
        if counter>=1:
            indexes[idx_array_indexes]=i
            idx_array_indexes+=1
            counter-=1
        counter+=ratio
    return indexes

def generate_meta_info_about_file(path_to_frames, path_to_label, frame_rate, needed_sample_rate, window_size, window_step):
    # считать все файлы
    # считать лейблы
    if path_to_frames=='D:\\Databases\\AffWild2\\frames_1\\107-30-640x480\\':
        a=1+2
    frames_filenames=np.array(os.listdir(path_to_frames))
    labels=pd.read_csv(path_to_label, header=None)
    timesteps=np.array([1./frame_rate*i for i in range(labels.shape[0])])
    # сравнить их количество
    if len(frames_filenames)!=labels.shape[0]:
        frames_filenames=frames_filenames[:labels.shape[0]]
    # проредить кадры и лэйблы
    thin_indexes=thin_array_indexes(frames_filenames, frame_rate, needed_sample_rate)
    frames_filenames=frames_filenames[thin_indexes]
    labels=labels.iloc[thin_indexes]
    timesteps=timesteps[thin_indexes]
    # сгенирировать индексы нарезки на окна
    cut_indexes=cut_sequence_on_windows(frames_filenames, window_size, window_step)
    # сгенерировать два массива нумпай - один - это файлнеймы картинок, второй - лэйблы и таймстепы
    result_data_filenames = np.zeros((cut_indexes.shape[0], window_size), dtype='<U9')
    result_labels_timesteps = np.zeros((cut_indexes.shape[0], window_size, labels.shape[-1] + 1), dtype='float32')
    if cut_indexes.shape[0]==1 and cut_indexes[0,1]>frames_filenames.shape[0]:
        result_data_filenames[:]=frames_filenames[-1]
        result_data_filenames[:,:frames_filenames.shape[0]]=frames_filenames
        result_labels_timesteps=result_labels_timesteps-1
        result_labels_timesteps[:,:labels.shape[0]] = np.concatenate((labels.values, timesteps.reshape((-1,1))), axis=-1)
    else:
        for i in range(cut_indexes.shape[0]):
            start, end= cut_indexes[i]
            result_data_filenames[i]=frames_filenames[start:end]
            result_labels_timesteps[i] = np.concatenate((labels.iloc[start:end].values, timesteps[start:end].reshape((-1,1))), axis=-1)
    # возвратить их из функции
    func = lambda x: path_to_frames + str(x)
    result_data_filenames = np.vectorize(func)(result_data_filenames)
    return result_data_filenames, result_labels_timesteps

def get_batch(general_data_metainfo, general_labels_timesteps_metainfo, shape_of_batch, start_idx, end_idx):
    batch_size=general_data_metainfo[start_idx:end_idx].shape[0]
    window_size=shape_of_batch[0]
    shapes_images=shape_of_batch[1:3]
    result_data_batch=np.zeros(shape=(batch_size,)+shape_of_batch)
    batch_idx=0
    for raw_idx in range(start_idx, end_idx, 1):
        for image_in_window_idx in range(window_size):
            img = PIL.Image.open(general_data_metainfo[raw_idx, image_in_window_idx])
            img = img.convert('RGB')
            img = img.resize(shapes_images)
            x = np.array(img)
            result_data_batch[batch_idx, image_in_window_idx] = x
    result_labels_timesteps_batch=general_labels_timesteps_metainfo[start_idx:end_idx]
    return result_data_batch, result_labels_timesteps_batch



def generate_and_save_batches(data_directory, path_to_labels, path_to_videos, path_to_output, window_size, window_step, final_frame_rate, batch_size,
                              shuffle=True, shapes_images=(224,224)):
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
    # считать файлнеймы лэйблов
    labels_filenames=os.listdir(path_to_labels)
    general_data_metainfo=[]
    general_labels_timesteps_metainfo=[]
    # цикл
    counter=0
    for labels_filename in labels_filenames:
        # читаем лэйбл и видео - находим frame rate
        video_filename=construct_video_filename_from_label(path_to_video=path_to_videos, label_filename=labels_filename)
        frame_rate=get_video_frame_rate(path_to_videos+video_filename)
        path_to_frames=data_directory+labels_filename.split('.')[0]+'\\'
        # генерируем метаинфу из прошлой функции
        meta_info_data, meta_info_labels_timesteps=generate_meta_info_about_file(path_to_frames=path_to_frames,
                                                                                 path_to_label=path_to_labels+labels_filename,
                                                                                 frame_rate=frame_rate,
                                                                                 needed_sample_rate=final_frame_rate,
                                                                                 window_size=window_size,
                                                                                 window_step=window_step)
        # сохраняем в заранее созданный лист
        general_data_metainfo.append(meta_info_data)
        general_labels_timesteps_metainfo.append(meta_info_labels_timesteps)
        print(counter, len(labels_filenames))
        counter+=1
    # закончить цикл
    # объединить лист с помощью vstack
    general_data_metainfo=np.vstack(general_data_metainfo)
    general_labels_timesteps_metainfo=np.vstack(general_labels_timesteps_metainfo)
    # перемешать, если нужно
    if shuffle==True:
        permutations=np.random.permutation(general_data_metainfo.shape[0])
        general_data_metainfo=general_data_metainfo[permutations]
        general_labels_timesteps_metainfo = general_labels_timesteps_metainfo[permutations]
    # цикл
    batch_num=0
    for batch_idx in range(0, general_data_metainfo.shape[0], batch_size):
        data, lbs_timesteps=get_batch(general_data_metainfo, general_labels_timesteps_metainfo,
                                      shape_of_batch=(window_size,)+shapes_images + (3,),
                                      start_idx=batch_idx, end_idx=batch_idx+batch_size)
        np.save(path_to_output + 'data_batch_num_%i' % batch_num, data.astype(np.uint8))
        np.save(path_to_output + 'labels_timesteps_batch_num_%i' % batch_num,
                lbs_timesteps)
        batch_num+=1


    '''    for window_idx in range(general_data_metainfo.shape[0]):
        images = np.zeros((window_size,) + shapes_images + (3,))
        for img_idx in range(images.shape[0]):
            img = PIL.Image.open(general_data_metainfo[window_idx, img_idx])
            img = img.convert('RGB')
            img = img.resize(shapes_images)
            x = np.array(img)
            images[img_idx]=x
        np.save(path_to_output + 'data_window_num_%i' % window_idx, images.astype(np.uint8))
        np.save(path_to_output + 'labels_timesteps_window_num_%i' % window_idx, general_labels_timesteps_metainfo[window_idx])
    '''
    # закончить цикл

def generate_validation_batches(data_directory, path_to_labels, path_to_videos, path_to_output, window_size, window_step, final_frame_rate, batch_size,
                              shapes_images=(224,224)):
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
    # считать файлнеймы лэйблов
    labels_filenames=os.listdir(path_to_labels)
    print(len(labels_filenames))
    for labels_filename in labels_filenames:
        # читаем лэйбл и видео - находим frame rate
        video_filename=construct_video_filename_from_label(path_to_video=path_to_videos, label_filename=labels_filename)
        frame_rate=get_video_frame_rate(path_to_videos+video_filename)
        path_to_frames=data_directory+labels_filename.split('.')[0]+'\\'
        # генерируем метаинфу из прошлой функции
        meta_info_data, meta_info_labels_timesteps=generate_meta_info_about_file(path_to_frames=path_to_frames,
                                                                                 path_to_label=path_to_labels+labels_filename,
                                                                                 frame_rate=frame_rate,
                                                                                 needed_sample_rate=final_frame_rate,
                                                                                 window_size=window_size,
                                                                                 window_step=window_step)
        meta_info_labels_timesteps_dataframe=form_dataframe_from_labels_timesteps_metainfo(meta_info_labels_timesteps, labels_filename)
        images=np.zeros(shape=(meta_info_data.shape[0], window_size)+shapes_images+(3,), dtype=np.uint8)
        for num_window in range(meta_info_data.shape[0]):
            for idx_window in range(window_size):
                img = PIL.Image.open(meta_info_data[num_window, idx_window])
                img = img.convert('RGB')
                img = img.resize(shapes_images)
                x = np.array(img)
                images[num_window, idx_window] = x
        num_batch=0
        for batch_idx in range(0, images.shape[0], batch_size):
            #TODO: check it one more time and also cutting on windows
            data_to_save=images[batch_idx:(batch_idx+batch_size)]
            labels_to_save=meta_info_labels_timesteps_dataframe.iloc[batch_idx*window_size:(batch_idx+batch_size)*window_size]
            np.save(path_to_output + labels_filename.split('.')[0]+'_data_batch_%i.npy'%(num_batch), data_to_save.astype(np.uint8))
            labels_to_save.to_csv(path_to_output + labels_filename.split('.')[0]+'_labels_batch_%i.csv'%(num_batch), index=False)
            num_batch+=1


def form_dataframe_from_labels_timesteps_metainfo(labels_timesteps_metainfo, filename):
    result_dataframe=pd.DataFrame(columns=['filename', 'valence', 'arousal', 'timestep'])

    for num_window in range(labels_timesteps_metainfo.shape[0]):
        df_tmp = pd.DataFrame(columns=['valence', 'arousal','timestep'], data=labels_timesteps_metainfo[num_window])
        df_tmp['filename'] = filename.split('.')[0]
        result_dataframe = pd.concat([result_dataframe,df_tmp])
    return result_dataframe



def main():
    '''add_missing_frames_in_all_folders(paths_to_folders='D:\\Databases\\AffWild2\\frames_1\\',
                                      path_to_labels='D:\\Databases\\AffWild2\\Annotations\\VA_Set\\validation\\Aligned_labels\\')'''
    generate_validation_batches(data_directory='D:\\Databases\\AffWild2\\frames_1\\',
                              path_to_labels='D:\\Databases\\AffWild2\\Annotations\\VA_Set\\validation\\\Aligned_labels\\',
                              path_to_videos='D:\\Databases\\AffWild2\\Videos\\',
                              path_to_output='D:\\Databases\\AffWild2\\batches_validation\\',
                              window_size=30,
                              window_step=30.*2./5.,
                              final_frame_rate=7.5,
                              batch_size=16,
                              shapes_images=(224, 224))



if __name__ == "__main__":
    main()