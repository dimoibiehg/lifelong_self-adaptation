from sklearn.multiclass import OneVsOneClassifier
from sklearn.datasets import load_svmlight_file
import numpy as np 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import random 
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
import plotly.graph_objects as go
import itertools
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, Adamax, Adadelta, Adagrad, SGD
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import initializers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
import math 
import sys, os
import keras_tuner as kt
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from lifelong_learner.task_manager import *
from tqdm import tqdm
import time
from self_adaptation.feedback_loop import *
from lifelong_learner.lifelong_learning_loop import *
from common.stream import Stream



random.seed(10)

def state_of_the_art_method_with_sub_batches():
    stream = Stream(13905)
    fl = FeedbackLoop(stream, None, operator_support=True)
    fl.offline_phase()
    for i in tqdm(range(13800)):
        fl.monitor()
        
    return fl.accs_baseline

def reference_method_with_sub_batches():
    stream = Stream(13905)
    fl = FeedbackLoop(stream, None)
    
    fl.offline_phase()
    for i in tqdm(range(13800)):
        fl.monitor()
        
    return fl.accs_baseline

def lifelong_learning_method_with_sub_batches():
    stream = Stream(13905)
    fl = FeedbackLoop(stream, None, operator_support=True)
    ll = LifelongLearningLoop(fl)
    fl.lifelong_loop = ll
    
    fl.offline_phase()
    ll.start()
    
    return fl.accs_baseline


def visualize_in_time(lists_of_classification, names, step_size = 100):
    
    data_plot = []
    data_dist_plot = []
    x = 60 * ["Adaptation cycle 1-6k"] + 40 * ["Adaptation cycle 6k - 10k"] + 37 * ["Adaptation cycle 10k - 13.7k"]
    for count, list_of_classification in enumerate(lists_of_classification):
        acc_batch = []
        for i in range(0, len(list_of_classification), step_size):
            acc_batch.append(sum(list_of_classification[i:(i+step_size)])/step_size)
        data_plot.append(go.Scatter(y = acc_batch[:-2], name=names[count]))
        data_dist_plot.append(go.Box(x = x, y = acc_batch[:-2], name=names[count], boxmean='sd'))
    fig = go.Figure(data = data_plot)
                        #go.Scatter(x = list(range(1, len(X)+1)), y = accs_drift_case, name="Effect of concept drift"),\
                        # go.Scatter(y = acc_batch[:-2], name="SVC offline training")])
                        # # go.Scatter(x = list(range(1, len(accs_baseline))), y = accs_baseline[0:-2], name="SVC using a previous batch"),
                        # # go.Scatter(x = list(range(1, len(accs_retrain_all))), y = accs_retrain_all[0:-2], name = "SVC using 5 previous batch"),
                        # # go.Scatter(x = list(range(1, len(accs_lifelong))), y = accs_lifelong[0:-2], name="SVC using a previous batch under LLL")])
    fig.update_layout(
            yaxis = dict(
                title = "Classification accuracy",
                range = [0, 1.3]
            ),
            xaxis = dict(
                title = 'Batch index (chronological order)',
                #tickvals = years_count[cYear]
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            boxmode='group',
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.6
            )
            # showlegend = False
        )           
    fig.show()
    
    
    fig = go.Figure(data = data_dist_plot)
    fig.update_layout(
            yaxis = dict(
                title = "Classification accuracy",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            boxmode='group',
            legend=dict(
                yanchor="top",
                y=0.15,
                xanchor="left",
                x=0.53
            )
        )           
    fig.show()

def visualize_drif_scanrio():
    stream = Stream(13900)
    # Represent inremental covariate drift  
    ch = np.array(stream.X_features)[np.where(np.array(stream.Y_features) == 1)[0]][:,0]

    means = [np.mean(ch[x:(x+400)]) for x in range(len(ch)-400)]
    fig = go.Figure(data = [go.Scatter(x = list(range(1, len(means)+1)), y = means, name="First channel of sensor 1 for gas type 1")])

    fig.update_layout(
            yaxis = dict(
                title = "Expected response of Sensor1-Channel1 for gas type 1 (ppmv)",
            ),
            xaxis = dict(
                title = 'Chronological index',
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )           
    fig.show()   


if __name__ == "__main__":
    visualize_drif_scanrio()  
    visualize_in_time([reference_method_with_sub_batches(), state_of_the_art_method_with_sub_batches(), lifelong_learning_method_with_sub_batches()], 
                  ["SVC offline training (Reference)", "SVC using a previous batch (State-of-the-art)", "SVC using a previous batch under LLL"])