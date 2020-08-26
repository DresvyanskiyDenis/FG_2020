from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd

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


    def calculate_FG_2020_categorical_score_across_all_instances(self, instances, delete_value=-1):
        # TODO: peredelat na bolee logichniy lad. Eto poka chto bistroo reshenie
        ground_truth_all=np.zeros((0,))
        predictions_all=np.zeros((0,))
        for instance in instances:
            ground_truth_all=np.concatenate((ground_truth_all, instance.labels))
            predictions_all = np.concatenate((predictions_all, instance.predictions))
        mask=ground_truth_all!=delete_value
        ground_truth_all=ground_truth_all[mask]
        predictions_all=predictions_all[mask]
        return f1_score(ground_truth_all, predictions_all, average='macro')


    def calculate_accuracy(self):
        return accuracy_score(self.ground_truth, self.predictions)

    def calculate_f1_score(self, mode='macro'):
        return f1_score(self.ground_truth, self.predictions, average=mode)

    def calculate_FG_2020_categorical_score(self, f1_score_mode='macro'):
        return 0.67*self.calculate_f1_score(f1_score_mode)+0.33*self.calculate_accuracy()