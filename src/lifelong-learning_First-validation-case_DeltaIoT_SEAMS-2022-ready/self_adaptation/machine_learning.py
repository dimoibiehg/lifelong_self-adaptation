from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from enum import Enum
from common.goal import *
from common.learning import *
from functools import partial

class MachineLearning:
    
    def __init__(self, \
                 list_of_learning_models = [{'model':SGDRegressor(loss='squared_epsilon_insensitive',penalty='elasticnet'),
                                             'type': LearnerType.SKLEARN,
                                             'scaler': StandardScaler(),
                                             'target_type': TargetType.LATENCY,
                                             'target_range': [0, 100]
                                            }],\
                 indices_of_active_models = [0]
                 ):
        self.list_of_learning_models = list_of_learning_models
        self.indices_of_active_models = indices_of_active_models

    def predict(self, x, learner_index):
        scaled_features = self.list_of_learning_models[learner_index]['scaler'].transform(x)
        if(self.list_of_learning_models[learner_index]['type'] == LearnerType.SKLEARN):
            target_type = self.list_of_learning_models[learner_index]['target_type']
            target_range = self.list_of_learning_models[learner_index]['target_range']
            prediction_result = np.array(self.list_of_learning_models[learner_index]['model'].predict(scaled_features))
            # adjust the prediction based on the tagret range (it is meaningful 
            # for computing the prediction error) 
            prediction_result = np.array(list(map(partial(self.__sign, a_range = target_range), prediction_result)))
            return (target_type, prediction_result)
        else:
            raise Exception('not implemented yet!')

    def continual_train(self, x, y, learner_index):
        self.list_of_learning_models[learner_index]['scaler'].partial_fit(x)
        scaler = self.list_of_learning_models[learner_index]['scaler']
        scaled_features = scaler.transform(x)
        if(self.list_of_learning_models[learner_index]['type'] == LearnerType.SKLEARN):
            return self.list_of_learning_models[learner_index]['model'].partial_fit(scaled_features, y)
        else:
            raise Exception('not implemented yet!')

    def train(self, x, y, learner_index):
        scaler = self.list_of_learning_models[learner_index]['scaler']
        scaler.fit(x)
        scaled_features = scaler.transform(x)
        if(self.list_of_learning_models[learner_index]['type'] == LearnerType.SKLEARN):
            return self.list_of_learning_models[learner_index]['model'].fit(scaled_features, y)
        else:
            raise Exception('not implemented yet!')


    def __sign(self, value, a_range):
        if((value >= a_range[0]) and (value <= a_range[1])):
            return value
        elif(value < a_range[0]):
            return a_range[0]
        else:
            return a_range[1]
