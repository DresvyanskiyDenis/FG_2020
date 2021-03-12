import numpy as np
import pandas as pd




if __name__=='__main__':
    load_path_data='C:\\Users\\Dresvyanskiy\\Downloads\\aff_wild2_val_emo_with_loss.pickle'
    load_path_labels='C:\\Users\\Dresvyanskiy\\Downloads\\df_affwild2_val_emo.csv'
    arr=np.load(load_path_data, allow_pickle=True)
    print(arr)
    print(arr.shape)
    print('##########################')
    labels=pd.read_csv(load_path_labels)
    print(labels.info())
    print(labels.shape)