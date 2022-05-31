from self_adaptation.feedback_loop import *
from common.helper import *

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, Adamax, Adadelta, Adagrad, SGD
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import initializers
from lifelong_self_adaptation.task_manager import *
from lifelong_self_adaptation.knowledge import *

from tqdm import tqdm
import keras_tuner as kt

from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn import metrics, model_selection
import sklearn.pipeline

# from incremental_trees.models.regression.streaming_rfr import StreamingRFR
from skmultiflow.trees import HoeffdingTreeRegressor, HoeffdingAdaptiveTreeRegressor

class LifelongLearningLoop:
    def __init__(self, feedbackloop, bilearning_mode = True, retrain_previous_models = True):
        self.feedback_loop = feedbackloop
        self.cycle_of_new_task_consideration = 100
        self.evolve_step = 20
        self.knowledge = KonwledgeStore()
        self.task_manager = TaskManager()
        self.max_instance_per_task = 5 * 216
        self.feature_size = 216
        self.cycle_id = 0
        self.bilearning_mode = bilearning_mode
        self.retrain_previous_models = retrain_previous_models
    def start(self):
        stream_len = self.feedback_loop.stream.cycles_num
        for i in tqdm(range(0, stream_len)):
            self.cycle_id = i
            if (i % self.evolve_step == 0 and i > 0):
                # first step: task manager called 
                a_new_task_detected, detected_task_id = self.task_manager.detect(self.knowledge.enivornment_feautres[-self.evolve_step:], apply_scale=False)
                self.knowledge_based_learner(a_new_task_detected, detected_task_id)

            self.feedback_loop.monitor()


    def task_based_miner(self, task_id):
        # extract task-specific data
        idxs = self.knowledge.task_ids[task_id]
        # this limit can be adjusted based on the corresponding task
        if(len(idxs) > self.max_instance_per_task):
            idxs = idxs[-self.max_instance_per_task:]
        features = [self.knowledge.selected_features[i] for i in idxs]
        targets = {}
        for target_type in TargetType:
            if target_type in self.knowledge.selected_targets:
                targets[target_type] =  [self.knowledge.selected_targets[target_type][i] for i in idxs]
        return features, targets


    def knowledge_based_learner(self, a_new_task_detected, detected_task_id):

        # update knowledge store
        if(a_new_task_detected):
            self.knowledge.task_ids.append([])
            detected_task_id = len(self.feedback_loop.learner_module.list_of_learning_models)//2
        end = int(self.cycle_id/self.evolve_step) * self.evolve_step * self.feature_size
        current_selected_feature_size = len(self.knowledge.selected_features)
        self.knowledge.task_ids[detected_task_id].extend(list(range(current_selected_feature_size - self.evolve_step, current_selected_feature_size)))     

        if(a_new_task_detected or self.retrain_previous_models):
            
            for order, am_idx in enumerate(self.feedback_loop.learner_module.indices_of_active_models):
                # evolve the feedback loop
                active_model_idx = am_idx
                active_model = self.feedback_loop.learner_module.list_of_learning_models[active_model_idx]
                target_type = active_model['target_type']
                target_range = active_model['target_range']

                # call task base knowledge miner to extract task-specific data
                features, targets = self.task_based_miner(detected_task_id)

                #process mined data
                scaler = StandardScaler()
                scaler.partial_fit(features)
                features = scaler.transform(features)
                targets = targets[target_type]

                blockPrint()

                tuner = kt.tuners.SklearnTuner(
                    oracle=kt.oracles.BayesianOptimizationOracle(
                        objective=kt.Objective('score', 'max'),
                        max_trials=60),
                    overwrite=True,
                    hypermodel=self.build_learning_model,
                    scoring=metrics.make_scorer(metrics.r2_score))
                tuner.search(np.array(features), np.array(targets))
                best_model = tuner.get_best_models(1)[0]
                best_model.partial_fit(features, targets)
                model_object = {'model':best_model,\
                                'type': LearnerType.SKLEARN,\
                                'scaler': scaler,\
                                'target_type': target_type,\
                                'target_range': target_range    
                            }

                enablePrint()
                if(a_new_task_detected):
                    print("******************New Task Detected*******************")
                    print("******************************************************")
                    print("******************************************************")
                    print("******************************************************")
                    print("******************************************************")
                    self.feedback_loop.learner_module.list_of_learning_models.append(model_object)
                else:
                    self.feedback_loop.learner_module.list_of_learning_models[detected_task_id * 2 + order] = model_object 

                print("////////////////// detected task id: %d" % (detected_task_id))
                self.feedback_loop.learner_module.indices_of_active_models = [detected_task_id * 2, detected_task_id *2 + 1]            


        with open(self.feedback_loop.out_base_addr + "task_ids.txt", "a+") as file_object:
            file_object.write(str(detected_task_id) + ",")
            
        with open(self.feedback_loop.out_base_addr + "learners_selection.txt", "a+") as file_object:
            
            learners_models = ""
            for order, am_idx in enumerate(self.feedback_loop.learner_module.indices_of_active_models):
                # evolve the feedback loop
                active_model_idx = am_idx
                active_model = self.feedback_loop.learner_module.list_of_learning_models[active_model_idx]
                if(str(active_model['model'])[0:15].startswith("SGD")):
                    learners_models += "SGD"
                elif(str(active_model['model'])[0:15].startswith("Hoef")):
                    learners_models += "HAT"
                learners_models += ","
                
            file_object.write(learners_models[0:-1] + "\n")
                    
    def build_learning_model(self, hp):
        if(self.bilearning_mode):
            return self.bi_learning_model(hp)
        return self.single_learning_model(hp)


    def bi_learning_model(self,hp):
        method_idx = hp.Choice("method_idx", [0,1])

        if(method_idx == 0):
            return SGDRegressor(loss = hp.Choice("sgd_loss", ['squared_loss', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive']),\
                                penalty=hp.Choice("sgd_penalty", ['l2', 'l1', 'elasticnet']))
        elif(method_idx == 1):
            return HoeffdingAdaptiveTreeRegressor(learning_ratio_perceptron = hp.Choice("learning_ratio_perceptron", [1e-5, 1e-4, 1e-3, 0.01, 0.1]))
        else:
            raise Exception("The learning method index is not in the range")
    
    def single_learning_model(self, hp):
        return SGDRegressor(loss = hp.Choice("sgd_loss", ['squared_loss', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive']),\
                                penalty=hp.Choice("sgd_penalty", ['l2', 'l1', 'elasticnet']))