from tqdm import tqdm
import keras_tuner as kt
import numpy as np
from lifelong_learner.knowledge import KonwledgeStore
from lifelong_learner.task_manager import TaskManager
from self_adaptation.feedback_loop import *
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from common.helper import *
from sklearn.metrics import accuracy_score, make_scorer
import sklearn.pipeline

class LifelongLearningLoop:
    def __init__(self, feedbackloop):
        self.feedback_loop = feedbackloop
        self.evolve_step = 20
        self.knowledge = KonwledgeStore()
        self.max_data_per_task = 1000 # for task-based knowledge miner based on VC dimension of SVC 
        self.task_manager = TaskManager()
        self.cycle_id = 0
        
    def start(self):
        stream_len = self.feedback_loop.stream.cycles_num
        self.task_manager.detect([x[0] for x in self.knowledge.data_pair[0:self.feedback_loop.operator_cycle]])
        self.knowledge.task_ids.append(list(range(self.feedback_loop.operator_cycle)))
        
        
        # it should be 13800 as 100 items have been used in offline phase of the feedback loop
        number_of_cycles = ((stream_len//self.feedback_loop.operator_cycle) - 1) * self.feedback_loop.operator_cycle
        for i in tqdm(range(0, number_of_cycles)):
            self.cycle_id = i
            if (i % self.evolve_step == 0 and i > 0):
                # first step: task manager called 
                a_new_task_detected, detected_task_id = self.task_manager.detect([x[0] for x in self.knowledge.data_pair[-self.evolve_step:]])
                # update knowledge store
                if(a_new_task_detected):
                    self.knowledge.task_ids.append([])
                    detected_task_id = len(self.knowledge.task_ids) - 1
                current_selected_feature_size = len(self.knowledge.data_pair)
                self.knowledge.task_ids[detected_task_id].extend(list(range(current_selected_feature_size - self.evolve_step, current_selected_feature_size)))  
                self.knowledge_based_learner(a_new_task_detected, detected_task_id)

            self.feedback_loop.monitor()


    def task_based_miner(self, task_id, is_new:bool):
        # extract task-specific data
        idxs = self.knowledge.task_ids[task_id]
        if((len(self.knowledge.data_pair) % self.feedback_loop.operator_cycle) == 0):
            self.knowledge.cache_current_flow_data_up_to_reach_operator_cycle = []
        elif(is_new):
            scaled_new_tasks_data = self.feedback_loop.scaler.transform([x[0] for x in self.knowledge.data_pair[-self.evolve_step:]])
            stds = [np.std(np.fabs(x)) for x in self.feedback_loop.calssifier_model.decision_function(scaled_new_tasks_data)]
            args_sorts = list(np.argsort(stds))
            dataX = [self.knowledge.data_pair[-self.evolve_step:][x][0] for x in args_sorts[0:5]]
            ## ask labels from operator
            number_of_passed_operator_cycle = 100 * (len(self.knowledge.data_pair)//100) 
            find_start_point = ((len(self.knowledge.data_pair) - number_of_passed_operator_cycle)//self.evolve_step) * self.evolve_step - self.evolve_step
            subset_Y =  self.feedback_loop.stream.Y_features[find_start_point:(find_start_point + self.evolve_step)]
            dataY = [subset_Y[x] for x in args_sorts[0:5]]
            for x in args_sorts[0:5]:
                self.knowledge.data_pair[-self.evolve_step:][x] = subset_Y[x]
            
            self.knowledge.cache_current_flow_data_up_to_reach_operator_cycle.extend(list(zip(dataX, dataY)))
        else:
            labeled_data = [self.knowledge.data_pair[id] for id in idxs if self.knowledge.data_pair[id][1] > -1]
            dataX= [x[0] for x in labeled_data]
            dataY= [x[1] for x in labeled_data]
            # sample limited number of simialr data for scalalbility
            s = list(range(len(dataY)))
            if(len(dataY) > self.max_data_per_task):                
                s = random.sample(s, self.max_data_per_task)
            self.knowledge.cache_current_flow_data_up_to_reach_operator_cycle.extend(list(zip([dataX[i] for i in s], [dataY[i] for i in s])))
        
        
        merge_all_required_data = self.knowledge.cache_previous_operator_fed_labeled_data + self.knowledge.cache_current_flow_data_up_to_reach_operator_cycle
        dataX = [x[0] for x in merge_all_required_data]
        dataY = [x[1] for x in merge_all_required_data]
        
        return  dataX, dataY   
        

    def knowledge_based_learner(self, a_new_task_detected, detected_task_id):

        dataX, dataY = self.task_based_miner(detected_task_id, a_new_task_detected)
        scaled_data =  self.feedback_loop.scaler.fit_transform(dataX)   
        self.feedback_loop.calssifier_model.fit(scaled_data, dataY)
        
        ## As the best parameter is given by the default value of gamma and C, 
        # we leave it as it is, without any optimization. You can uncomment the following lines, and 
        # comment the above line. Note that, there is also an issue with Keras tuner for this part of optimization 
        # as mentioned here: https://github.com/keras-team/keras-tuner/issues/394
        # you need to fix the bug in the soruce of the keras tuner library.
        
        # blockPrint()

        # tuner = kt.tuners.SklearnTuner(
        #     oracle=kt.oracles.BayesianOptimizationOracle(
        #         objective=kt.Objective('score', 'max'),
        #         max_trials=60),
        #     overwrite=True,
        #     hypermodel=self.build_learning_model,
        #     scoring=make_scorer(accuracy_score))
        # # print("***********************")
        # # print(np.array(features).shape)
        # # print(np.array(targets).shape)
        # # print("-----------------------")
        # tuner.search(np.array(scaled_data), np.array(dataY))
        # best_model = tuner.get_best_models(1)[0]
        # best_model.fit(scaled_data, dataY)
        
        # enablePrint()     
        # self.feedback_loop.calssifier_model = best_model
    
    def build_learning_model(self, hp):
        return OneVsOneClassifier(SVC(random_state=0,kernel='rbf', C = hp.Choice("C", [float(2**i) for i in range(-5, 10)]), 
                                                                gamma = hp.Choice("gamma", [float(2**i) for i in range(-10, 5)])))