# from common.helper import *
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, Adamax, Adadelta, Adagrad, SGD
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import initializers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from functools import partial
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras_tuner as kt
import scipy.stats as stats
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import mannwhitneyu
import os
import sys
import logging
from common.helper import *

tf.get_logger().setLevel(logging.ERROR)


class TaskManager:
    def __init__(self):
        self.count_search = 0 
        self.training_percent = 80
        self.task_encoders = []
        self.task_encodings = [] # can be link to knowledge store (here, becasue of efficiency)
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.SIGNIFICANCE = 0.05


    def detect(self, x, apply_scale = True):
        a_new_task_detected = False
        task_id = 0
        x = np.array(x)
        if(apply_scale):
            self.scaler.partial_fit(x)
        scaled_x = self.scaler.transform(x)
        if(len(self.task_encoders) == 0): #first coming task to the system (initilization)
            blockPrint()

            best_model = self.find_best_auto_encoder(scaled_x)

            input_dim = len(x[0])
            best_model.fit(scaled_x, scaled_x,
                        batch_size=len(x),
                        epochs=30,
                        verbose=False)
            self.task_encoders.append(best_model)
            self.task_encodings.append(list(x))
            enablePrint()
        else:
            encodres_dist = []
            arg_min_ind = -1
            pvalues = []
            for i in range(len(self.task_encoders)):
                encoder = self.task_encoders[i]
                score = encoder.predict(scaled_x).tolist()
                score_dist = [np.sqrt(np.sum((np.subtract(scaled_x[j],score[j]))**2)) for j in range(len(scaled_x))]
                score_dist_accum = [np.mean(score_dist[j:(j+5)]) for j in range(0, len(score_dist)-5 + 1, 5)]
                
                new_encodings = encoder.predict(np.array(self.task_encodings[i])).tolist()
                new_encodings_dist = [np.sqrt(np.sum((np.subtract(new_encodings[j],self.task_encodings[i][j]))**2)) for j in range(len(new_encodings))]
                new_encodings_dist_accum = [np.mean(new_encodings_dist[j:(j+5)]) for j in range(0, len(new_encodings_dist)-5+1, 5)]
                U1, p = mannwhitneyu(score_dist_accum, new_encodings_dist_accum, use_continuity=True, alternative='two-sided')
                pvalue = p
                pvalues.append(pvalue)
            
            max_pvalue_idx = np.argmax(pvalues)
            sorted_pvalues = sorted(pvalues)
            #Holm's method
            corrected_pvals = [int((pval <= (self.SIGNIFICANCE/(len(sorted_pvalues) - idx_pval)))) 
                               for (idx_pval, pval) in enumerate(sorted_pvalues)]
            reject = (sum(corrected_pvals) > 0)
            if(~reject):
                arg_min_ind = max_pvalue_idx

            if(arg_min_ind > -1):
                blockPrint()
                scaled_x = x # self.scaler[arg_min_ind].transform(x)
                self.task_encoders[arg_min_ind].fit(scaled_x, scaled_x)
                encoder = self.task_encoders[arg_min_ind]
                encoder = self.extract_specific_layer(encoder)
                self.task_encodings[arg_min_ind].extend(x.tolist())
                task_id = arg_min_ind
                enablePrint()
            else: # not similar to any previously detected task
                print("############### detected new task ###############")
                blockPrint()
                scaled_x = x 

                best_model = self.find_best_auto_encoder(scaled_x)
                input_dim = len(x[0])
                best_model.fit(scaled_x, scaled_x,
                        batch_size=len(x),
                        epochs=60,
                        verbose=False)
                enablePrint()
                self.task_encoders.append(best_model)
                encoder = self.extract_specific_layer(best_model)
                self.task_encodings.append(x.tolist())
                a_new_task_detected = True
                task_id = -1


        return a_new_task_detected, task_id


    def find_best_auto_encoder(self, x):
        blockPrint()

        input_dim = len(x[0])
        # powe_2 = highestPowerof2(input_dim)
        model_creator = partial(self.build_auto_encoder_model, input_dim = input_dim)
        tuner = kt.BayesianOptimization(
            model_creator, 
            objective='val_loss', 
            max_trials = 30,
            overwrite= True,
            project_name="lifelong_with_considering_per_task_0001_with_top_1_uncertainties")

        tuner.search(x,x, epochs=30, validation_split=0.2, verbose=0)
        best_model = tuner.get_best_models(1)[0]

        enablePrint()
        return best_model

    def extract_specific_layer(self, original_model, layer_name = "encoder"):
        encoder = Model(inputs=original_model.input,
                        outputs=original_model.get_layer(layer_name).output)
        return encoder

    def build_auto_encoder_model(self, hp, input_dim):
        model = Sequential()

        regulizer_rate = hp.Choice("regulizer_rate", [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        
        if(input_dim < 30): # for low dimension space we do not need to too much layers
            model.add(Dense(input_dim - 2, input_dim = input_dim, activation='relu', kernel_regularizer=l1(regulizer_rate)))
            model.add(Dense(input_dim - 4, name = "encoder", activation='sigmoid', kernel_regularizer=l1(regulizer_rate)))
            model.add(Dense(input_dim - 2, activation='relu', kernel_regularizer=l1(regulizer_rate)))
        else:
            minimum_difference_between_layers = 2
            endocer_size = hp.Choice("encoder_size", [int(input_dim/2), int(input_dim/4), int(input_dim/8), int(input_dim/16)])
            layer_steps = hp.Choice("layer_steps", [int(input_dim/2), int(input_dim/4), int(input_dim/8), int(input_dim/16)])

            current_layer_size = input_dim - layer_steps
            model.add(Dense(current_layer_size, input_dim = input_dim, activation='relu', kernel_regularizer=l1(regulizer_rate)))

            while (current_layer_size - layer_steps) > (endocer_size + minimum_difference_between_layers):
                current_layer_size -= layer_steps
                model.add(Dense(current_layer_size, activation='relu', kernel_regularizer=l1(regulizer_rate)))


            model.add(Dense(endocer_size, name = "encoder", activation='sigmoid', kernel_regularizer=l1(regulizer_rate)))

            if(current_layer_size < input_dim):
                model.add(Dense(current_layer_size,activation='relu', kernel_regularizer=l1(regulizer_rate)))


            while ((current_layer_size + layer_steps) + minimum_difference_between_layers)< input_dim:
                current_layer_size += layer_steps
                model.add(Dense(current_layer_size, activation='relu', kernel_regularizer=l1(regulizer_rate)))

        model.add(Dense(input_dim, activation='relu', kernel_regularizer=l1(regulizer_rate)))

        model.compile(optimizer=Adam(hp.Choice("learning_rate", [1e-1, 1e-2, 1e-3,1e-4,1e-5])), loss='mse')

        return model



    def kl_divergence(self, P,Q):
        epsilon = 0.00001

        # You may want to instead make copies to avoid changing the np arrays.
        P = P+epsilon
        Q = Q+epsilon

        divergence = np.sum(P*np.log(P/Q))
        return divergence

    def hellinger(self, P, Q):
        avg_P = np.average(P,axis=0)
        avg_Q = np.average(Q,axis=0)
        h = np.sqrt(np.sum((np.sqrt(avg_P) - np.sqrt(avg_Q))**2))/np.sqrt(2)
        return h

    def mst_edges(self,V, k):
        """
        Construct the approximate minimum spanning tree from vectors V
        :param: V: 2D array, sequence of vectors
        :param: k: int the number of neighbor to consider for each vector
        :return: V ndarray of edges forming the MST
        """

        k = min(len(V) - 1, k)

        # generate a sparse graph using the k nearest neighbors of each point
        G = kneighbors_graph(V, n_neighbors=k, mode='distance')

        # Compute the minimum spanning tree of this graph
        full_tree = minimum_spanning_tree(G, overwrite=True)

        return np.array(full_tree.nonzero()).T