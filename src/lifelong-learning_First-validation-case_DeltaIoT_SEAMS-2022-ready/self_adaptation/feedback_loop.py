from common.stream import *
import numpy as np
from self_adaptation.machine_learning import *
from enum import Enum
from abc import ABC
from functools import reduce
import random
from common.goal import *
from pathlib import Path
import pickle
from itertools import chain
import time

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SPXCrossover, BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem import ZDT1
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.lab.visualization.plotting import Plot


class FeedbackLoop:
    def __init__(self, stream, learner_module, goals, lifelong_loop = None, out_base_addr = "./data/predicts/", retrain = False, is_utility_based = False):
        self.stream = stream
        self.lifelong_loop = lifelong_loop
        self.out_base_addr = out_base_addr
        self.collected_data = {'features':[]}
        self.learner_module = learner_module
        self.goals = goals
        self.features = None 
        self.targets = None
        self.retrain = retrain
        self.is_utility_based = is_utility_based

    @property
    def features(self):
        return self.__features 

    @features.setter
    def features(self, value):
        self.__features = value
        if((self.lifelong_loop != None) and (value is not None)):
            self.lifelong_loop.knowledge.enivornment_feautres.append(value[0][0:17])


    @property
    def targets(self):
        return self.__targets

    @targets.setter
    def targets(self, value):
        self.__targets = value

    def max_expected_utility_arg(self, predicts):
        pl_vals = []
        ec_vals = []
        for goal in self.goals:
            if(goal.target_type == TargetType.PACKETLOSS):
                pl_vals =  predicts[goal.target_type]
            elif(goal.target_type == TargetType.ENERGY_CONSUMPTION):
                ec_vals =  predicts[goal.target_type]
        all_goal_vals = [weight_for_targte_type(TargetType.PACKETLOSS) * utility_fucntion(TargetType.PACKETLOSS, x) + \
                         weight_for_targte_type(TargetType.ENERGY_CONSUMPTION) * utility_fucntion(TargetType.ENERGY_CONSUMPTION, y) \
                         for (x,y) in zip(pl_vals, ec_vals)]
        
        return np.argmax(all_goal_vals)
        

    def monitor(self):
        self.features, self.targets = self.stream.read_current_cycle()
        self.analyse_and_plan()
    
    def analyse_and_plan(self):
        # just one option is used for monitoring and others collected in the matter of validation 
        selected_option_id = 0
        if(not self.is_utility_based):
            if(self.stream.current_cyle > 2):
                satisfied_goals_idxs = []
                all_goals_optimization = True
                for goal in self.goals:
                    if(goal.type is not GoalType.OPTIMIZATION):
                        all_goals_optimization = False
                        break
                
                ## supposed that only one goal is defined over each quality property
                if(all_goals_optimization and len(self.goals) > 1): # a multi-objective optimization case
                    predicts = {}
                    if((len(self.goals) == 2) or len(self.goals) == 3):
                        for goal in self.goals:
                            for i in self.learner_module.indices_of_active_models:
                                    if(self.learner_module.list_of_learning_models[i]['target_type'] == goal.target_type):
                                        _, predicts[goal.target_type] = self.learner_module.predict(self.features, i)
                                        break
                        if(len(self.goals) == 2):
                            problem = DeltaIoTTwoObjective(predicts[self.goals[0].target_type], 
                                                        predicts[self.goals[1].target_type],
                                                        TargetRange[str(self.goals[0].target_type)].value[1],
                                                        TargetRange[str(self.goals[1].target_type)].value[1])
                        else:
                            problem = DeltaIoTMultiObjective(predicts[TargetType.PACKETLOSS], predicts[TargetType.ENERGY_CONSUMPTION], predicts[TargetType.LATENCY])
                        algorithm = NSGAII(
                                        problem=problem,
                                        population_size=100,
                                        offspring_population_size=100,
                                        mutation=BitFlipMutation(probability=1.0),
                                        crossover=SPXCrossover(probability=1.0),
                                        termination_criterion=StoppingByEvaluations(max_evaluations=5000)
                                    )

                        algorithm.run()
                        
                        solutions = algorithm.get_result()
                        front = get_non_dominated_solutions(solutions)
                        ideal_point = [0] * len(self.goals)
                        min_point_dist = math.inf
                        min_point_idx = -1
                        for i in range(len(front)):
                            idx = front[i].variables[0].index(True)
                            if(len(self.goals) == 2):
                                pareto_point = [predicts[self.goals[0].target_type][idx],\
                                                predicts[self.goals[1].target_type][idx]]
                            else:
                                pareto_point = [predicts[TargetType.PACKETLOSS][idx],\
                                                predicts[TargetType.ENERGY_CONSUMPTION][idx],\
                                                predicts[TargetType.LATENCY][idx]]
                            to_ideal_dist = np.sum(np.subtract(pareto_point, ideal_point)**2)
                            # to_ideal_dist = pareto_point[0]
                            if(to_ideal_dist < min_point_dist):
                                min_point_dist = to_ideal_dist
                                min_point_idx = idx
                                
                        selected_option_id = min_point_idx
                        
                        Path(self.out_base_addr).mkdir(parents=True, exist_ok=True)
                        with open(self.out_base_addr + "cycle" + str(self.stream.current_cyle-1) + ".pkl", "wb") as outfile:
                            pickle.dump((predicts, selected_option_id), outfile)     
                        
                    else:
                        raise NotImplementedError("number of optimization goals not adequatte for this validation. Here, it should be 3.")
                else:
                    predicts = {}
                    addrs = {}
                    for goal in self.goals:
                        addrs[goal.target_type] = self.out_base_addr + (str(goal.target_type)) + "/" 
                        
                        
                        for i in self.learner_module.indices_of_active_models:
                            if(self.learner_module.list_of_learning_models[i]['target_type'] == goal.target_type):
                                _, predicts[goal.target_type] = self.learner_module.predict(self.features, i)
                                break
                        if(goal.type == GoalType.THRESHOLD):
                            if(goal.compare_type == CompareType.GREATER):
                                satisfied_goals_idxs.append(np.where(predicts[goal.target_type] > goal.value))
                            elif(goal.compare_type == CompareType.LESS):
                                satisfied_goals_idxs.append(np.where(predicts[goal.target_type] < goal.value))                
                            else:
                                raise Exception("extra goal compare type not supported yet!")

                        ## by default find the minimum 
                        ## for maximum you need to negative all values
                        elif(goal.type == GoalType.OPTIMIZATION):
                            pass


                        else:
                            raise Exception("extra goal type not implemented yet!")
                    
                    all_goals_satisfied_idxs = []
                    if(len(satisfied_goals_idxs) > 1):
                        all_goals_satisfied_idxs =  reduce(np.intersect1d, satisfied_goals_idxs)
                    elif(len(satisfied_goals_idxs) == 1):
                        if(any(isinstance(el, list) for el in all_goals_satisfied_idxs)):
                            all_goals_satisfied_idxs = all_goals_satisfied_idxs[0]
                    
                    for goal in self.goals:
                        if(goal.type == GoalType.OPTIMIZATION):
                            vals = [predicts[goal.target_type][x] for x in all_goals_satisfied_idxs]
                            if(goal.optimization_type == OptimizationType.MIN):
                                if(len(vals) > 0):
                                    all_goals_satisfied_idxs = [all_goals_satisfied_idxs[np.argmin(vals)]]
                                else:
                                    all_goals_satisfied_idxs = [np.argmin(predicts[goal.target_type])]
                            elif(goal.optimization_type == OptimizationType.Max):
                                if(len(vals) > 0):
                                    all_goals_satisfied_idxs = [all_goals_satisfied_idxs[np.argmax(vals)]]
                                else:
                                    all_goals_satisfied_idxs = [np.argmax(predicts[goal.target_type])]
                            else:
                                raise Exception("goal optimization type is not correct!")
                            break
                    
                    if(len(all_goals_satisfied_idxs) > 0):
                        if(any(isinstance(el, list) for el in all_goals_satisfied_idxs)):
                            all_goals_satisfied_idxs = all_goals_satisfied_idxs[0]
                        random.shuffle(all_goals_satisfied_idxs)
                        selected_option_id = all_goals_satisfied_idxs[0]
                        
                    else:
                        print("Default option has been selected.")

                    addrs_values = list(addrs.values())
                    Path(addrs_values[0]).mkdir(parents=True, exist_ok=True)
                    with open(addrs_values[0] + "cycle" + str(self.stream.current_cyle-1) + ".pkl", "wb") as outfile:
                        pickle.dump((predicts, selected_option_id), outfile)        
            
        else:
            if(self.stream.current_cyle > 2):
                predicts = {}
                addrs = {}
                for goal in self.goals:
                    addrs[goal.target_type] = self.out_base_addr + (str(goal.target_type)) + "/" 
                    for i in self.learner_module.indices_of_active_models:
                        if(self.learner_module.list_of_learning_models[i]['target_type'] == goal.target_type):
                            _, predicts[goal.target_type] = self.learner_module.predict(self.features, i)
                            break
                
                selected_option_id = self.max_expected_utility_arg(predicts)
                
                addrs_values = list(addrs.values())
                Path(addrs_values[0]).mkdir(parents=True, exist_ok=True)
                with open(addrs_values[0] + "cycle" + str(self.stream.current_cyle-1) + ".pkl", "wb") as outfile:
                    pickle.dump((predicts, selected_option_id), outfile)
                
                
        ################ Training Stage ###################
        features = self.features[selected_option_id]
        if(self.lifelong_loop != None):
            self.lifelong_loop.knowledge.selected_features.append(features)
            for goal in self.goals:
                if(goal.target_type in self.lifelong_loop.knowledge.selected_targets):
                    pass
                else:    
                    self.lifelong_loop.knowledge.selected_targets[goal.target_type] = []

                self.lifelong_loop.knowledge.selected_targets[goal.target_type].append(self.targets[goal.target_type][selected_option_id])

        
        self.collected_data['features'].append(features)
        for goal in self.goals:
            target = self.targets[goal.target_type][selected_option_id]
            if goal.target_type in self.collected_data:
                self.collected_data[goal.target_type].append(target)
            else:
                self.collected_data[goal.target_type] = [target]

        training_time = 0
        for i in self.learner_module.indices_of_active_models:
            start = time.time()
            if(self.retrain):       
                self.learner_module.train(self.collected_data['features'],\
                                          self.collected_data[self.learner_module.list_of_learning_models[i]['target_type']], i)
            else:
                self.learner_module.continual_train([features],\
                                                    [self.targets[self.learner_module.list_of_learning_models[i]['target_type']][selected_option_id]],\
                                                    i)
            end = time.time()
            training_time += (end - start)
         
        # collecting training time    
        if(self.stream.current_cyle > 2):
            with open(addrs_values[0] + "training_time.txt", "a+") as file_object:
                file_object.write(str(training_time) + ",")
            
        self.execute()

    def execute(self):
        pass


    def find_best_optimization_option():
        pass