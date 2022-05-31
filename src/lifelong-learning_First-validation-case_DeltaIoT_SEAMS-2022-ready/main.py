from self_adaptation.machine_learning import *
from self_adaptation.feedback_loop import *
from common.goal import *
from common.stream import *
from visualizer.visualizer import *
import random
from lifelong_self_adaptation.task_manager import *
from lifelong_self_adaptation.lifelong_learning_loop import *
import plotly.figure_factory as ff
from scipy.spatial.distance import euclidean
import math
from scipy.stats import mannwhitneyu
from pathlib import Path

import itertools
import plotly.express as px

def fetch_stream(stream_len, with_drift = True, target_types = [TargetType.LATENCY, TargetType.PACKETLOSS, TargetType.ENERGY_CONSUMPTION],
                                                target_names = ["latency", "packetloss", "energyconsumption"],):
    if(with_drift):
        return Stream(stream_len, target_types= target_types,\
                    target_names= target_names, \
                    stream_addr = "./data/<input for drift scenario data - obtained from a deltaiot simulation ")
    
    return Stream(stream_len, target_types= target_types,\
                    target_names= target_names, \
                    stream_addr = "input for no drift scenario data - obtained from a deltaiot simulation")
    
def run_feedbackloop():
    stream_len = 1500
    stream = fetch_stream(stream_len, with_drift=True) 
    learner = MachineLearning(list_of_learning_models = [
                                           {'model':SGDRegressor(loss='squared_epsilon_insensitive',penalty='elasticnet'),
                                             'type': LearnerType.SKLEARN,
                                             'scaler': StandardScaler(),
                                             'target_type': TargetType.PACKETLOSS,
                                             'target_range': TargetRange.PACKETLOSS.value
                                            },
                                            {'model':SGDRegressor(loss='squared_epsilon_insensitive',penalty='elasticnet'),
                                             'type': LearnerType.SKLEARN,
                                             'scaler': StandardScaler(),
                                             'target_type': TargetType.ENERGY_CONSUMPTION,
                                             'target_range': TargetRange.ENERGY_CONSUMPTION.value
                                            }
                                            ],
                                            indices_of_active_models = [0, 1]#, 2]
                                            )
    feedbackloop = FeedbackLoop(stream, learner,\
                                [
                                # Here it does not matter which type of goal is selected, 
                                # as the current scenario is a utiltiy-based goal between packet loss and energy consumption
                                OptimizationGoal(TargetType.PACKETLOSS, OptimizationType.MIN), 
                                ThrehsholdGoal(TargetType.ENERGY_CONSUMPTION, CompareType.LESS, 13.15)
                                ],\
                                retrain = True,    
                                is_utility_based = True,
                            out_base_addr="./data/<name of scenario>/")
                            # examples of <name of scenario> are, 
                            # "with_drift_indermental_SGD_with_LLL", or
                            # "without_drift_incremental_SGC_wihtout_LLL"
                            # By the way, it can be any other desired validname 
                                
    for i in tqdm(range(stream_len)):
        feedbackloop.monitor()

def run_ll_loop():
    stream_len = 1500
    target_type = TargetType.PACKETLOSS
    target_name= TargetName.PACKETLOSS
    targte_range = TargetRange.PACKETLOSS.value

    stream = fetch_stream(stream_len, with_drift=False)
    
    learner = MachineLearning(list_of_learning_models = [
                                           {'model':SGDRegressor(loss='squared_epsilon_insensitive',penalty='elasticnet'),
                                             'type': LearnerType.SKLEARN,
                                             'scaler': StandardScaler(),
                                             'target_type': TargetType.PACKETLOSS,
                                             'target_range': TargetRange.PACKETLOSS.value
                                            },
                                            {'model':SGDRegressor(loss='squared_epsilon_insensitive',penalty='elasticnet'),
                                             'type': LearnerType.SKLEARN,
                                             'scaler': StandardScaler(),
                                             'target_type': TargetType.ENERGY_CONSUMPTION,
                                             'target_range': TargetRange.ENERGY_CONSUMPTION.value
                                            }
                                            ],
                                            indices_of_active_models = [0, 1]
                                            )
    feedbackloop = FeedbackLoop(stream, learner,\
                                [
                                OptimizationGoal(TargetType.PACKETLOSS, OptimizationType.MIN), 
                                ThrehsholdGoal(TargetType.ENERGY_CONSUMPTION, CompareType.LESS, 13.15)
                                # OptimizationGoal(TargetType.LATENCY, OptimizationType.MIN)
                                ],\
                                is_utility_based = True,
                            out_base_addr="./data/<address of output>/")
                            
    
    ll = LifelongLearningLoop(feedbackloop, bilearning_mode=True, retrain_previous_models=True)
    ll.feedback_loop.lifelong_loop = ll
    ll.start()

def qos_visualization_multiple_goals_with_utility():
    def compute_utility(pl, ec):
        return weight_for_targte_type(TargetType.PACKETLOSS) * utility_fucntion(TargetType.PACKETLOSS, pl) + \
               weight_for_targte_type(TargetType.ENERGY_CONSUMPTION) * utility_fucntion(TargetType.ENERGY_CONSUMPTION, ec)
    def find_the_best_optimization_option(stream, base_addr, best_in_total = False):
        stream.read_current_cycle()
        qos_vals = []
        for i in tqdm(range(1, stream.cycles_num)):
            
            # reading actual values
            features, targets = stream.read_current_cycle()
            targets_pl = targets[TargetType.PACKETLOSS]
            targets_la = targets[TargetType.LATENCY]
            targets_ec = targets[TargetType.ENERGY_CONSUMPTION]

            #reading predicted values
            predicts, selected_option_id = pickle.load(open(base_addr +\
                               "cycle" + str(stream.current_cyle - 1) + ".pkl", "rb"))
            ## todo: can be generalized, works only for one optimization goal
            if(best_in_total):
                all_goal_vals = [compute_utility(x,y) for (x,y) in zip(targets_pl, targets_ec)]
                
                idx = np.argmax(all_goal_vals)
                qos_vals.append([targets_pl[idx], targets_ec[idx], all_goal_vals[idx]])
            else:
                qos_vals.append([targets_pl[selected_option_id], targets_ec[selected_option_id], compute_utility(targets_pl[selected_option_id], targets_ec[selected_option_id])])
        
        return qos_vals

    stream_len = 1500
    
    stream_with = fetch_stream(stream_len, with_drift=True)
    stream_optimal_CD = fetch_stream(stream_len, with_drift=True)
    stream_with_retrain = fetch_stream(stream_len, with_drift=True)
    stream_without = fetch_stream(stream_len, with_drift=False)
    stream_without_retrain = fetch_stream(stream_len, with_drift=False)
    stream_optimal = fetch_stream(stream_len, with_drift=False)
    stream_with_ll_SGD_RF = fetch_stream(stream_len, with_drift=True)
    stream_with_ll_only_SGD = fetch_stream(stream_len, with_drift=True)
    stream_without_ll_only_SGD = fetch_stream(stream_len, with_drift=False)
    stream_without_ll_SGD_RF = fetch_stream(stream_len, with_drift=False)  
    
    
    qos_without = find_the_best_optimization_option(stream_without,"./data/<path to the generated file>/")
    
    qos_with = find_the_best_optimization_option(stream_with,"./data/<path to the generated file>/")
    
    qos_without_retrain = find_the_best_optimization_option(stream_without_retrain,"./data<path to the generated file>/")
    qos_with_retrain = find_the_best_optimization_option(stream_with_retrain,"./data/<path to the generated file>/")
    
    qos_with_ll_SGD_RF = find_the_best_optimization_option(stream_with_ll_SGD_RF,"./data/<path to the generated file>/")
    qos_without_ll_only_SGD = find_the_best_optimization_option(stream_without_ll_only_SGD,"./data/<path to the generated file>/")
    qos_with_ll_only_SGD = find_the_best_optimization_option(stream_with_ll_only_SGD,"./data/<path to the generated file>/")
    
    qos_without_ll_SGD_RF = find_the_best_optimization_option(stream_without_ll_SGD_RF,"./data/<path to the generated file>/")
    qos_total_CD = find_the_best_optimization_option(stream_optimal_CD,"./data/<path to the generated file>/", best_in_total= True)

    qos_total = find_the_best_optimization_option(stream_optimal,"./data/<path to the generated file>/", best_in_total= True)

    
    qos_without_error = [np.subtract(qos_without[i], qos_total[i]).tolist() for i in range(len(qos_without))]
    qos_with_error = [np.subtract(qos_with[i], qos_total_CD[i]).tolist() for i in range(len(qos_with))]
    qos_without_retrain_error = [np.subtract(qos_without_retrain[i], qos_total[i]).tolist() for i in range(len(qos_without))]
    qos_with_retrain_error = [np.subtract(qos_with_retrain[i], qos_total_CD[i]).tolist() for i in range(len(qos_with))]

    qos_with_ll_error = [np.subtract(qos_with_ll_SGD_RF[i], qos_total_CD[i]).tolist() for i in range(len(qos_with))]
    qos_without_ll_only_SGD_error = [np.subtract(qos_without_ll_only_SGD[i], qos_total[i]).tolist() for i in range(len(qos_with))]
    qos_with_ll_only_SGD_error = [np.subtract(qos_with_ll_only_SGD[i], qos_total_CD[i]).tolist() for i in range(len(qos_with))]
    qos_without_ll_error = [np.subtract(qos_without_ll_SGD_RF[i], qos_total[i]).tolist() for i in range(len(qos_with))]
    
    plot_metrics([[x[0] for x in qos_without], [x[0] for x in qos_total]],\
                 ["ML-based best option", "True best option"],\
                 stream_len, [[300,300],[600,600],[900,900],[1200,1200]], [[-100,100],[-100,100],[-100,100],[-100,100]],\
                 "Packet Loss (%)", "pl_compare_best_and_ml_without_drift_over_time", "No drift occured", y_range = [0,100], add_vertical=False)
    
    plot_metrics([[x[0] for x in qos_with], [x[0] for x in qos_total_CD]],\
                 ["ML-based best option", "True best option"],\
                 stream_len, [[0,0], [100,100],[350,350],[720,720],[1070,1070], [1500, 1500]], [[-100,100],[-100,100],[-100,100],[-100,100]],\
                 "Packet Loss (%)", "pl_compare_best_and_ml_with_drift_over_time", "Concept drift occured", y_range = [0,100])
    

    group_names = len(qos_without_error) * ["No drift"] + len(qos_with_error) * ["Under drift"]
    plot_distributions([group_names], [[x[0] for x in qos_without_error] + [x[0] for x in qos_with_error]], ["SGD wihtout LLL"], ["gray"], 
                    "Different scenarios", "Signed packet loss error (%)", 0.03, 0.99, 
                    is_group = True,file_name = "pl_compare_with_and_without_drift_problem_distirbution",
                    )

    plot_distributions([group_names], [[x[1] for x in qos_without_error] + [x[1] for x in qos_with_error]], ["SGD wihtout LLL"], ["gray"], 
                    "Different scenarios", "Signed energy consumption error (mC)", 0.03, 0.99, 
                    is_group = True,file_name = "ec_compare_with_and_without_drift_problem_distirbution",
                    )
    
    plot_distributions([group_names], [[x[2] for x in qos_without_error] + [x[2] for x in qos_with_error]], ["SGD wihtout LLL"], ["gray"], 
                    "Different scenarios", "Signed expected utility error", 0.03, 0.99, 
                    is_group = True,file_name = "eu_compare_with_and_without_drift_problem_distirbution",
                    )
    
    baseline_name = "Baseline"
    state_of_the_art_name = "State-of-the-art"
    incremental_no_LLL = "SGD without LLL"
    lll_sgd_hat = "SGD + HAT with LLL"
    lll_sgd = "SGD with LLL "
    
    
    y = [[x[0] for x in qos_total] + [x[0] for x in qos_total_CD],
         [x[0] for x in qos_without_retrain] + [x[0] for x in qos_with_retrain],
         [x[0] for x in qos_without] + [x[0] for x in qos_with],
         [x[0] for x in qos_without_ll_only_SGD] + [x[0] for x in qos_with_ll_only_SGD],
         [x[0] for x in qos_without_ll_SGD_RF] + [x[0] for x in qos_with_ll_SGD_RF]
         ]
    names = [baseline_name, state_of_the_art_name, incremental_no_LLL, lll_sgd, lll_sgd_hat]
    colors = ["#E15F99", "lightseagreen", "gray", "rgb(127, 60, 141)", "rgb(175, 100, 88)"]
    plot_distributions([group_names] * len(y), y, names, colors, 
                       "Different scenarios", "Packet loss (%)", 0.06, 0.99, 
                       is_group = True,file_name = "pl_compare_with_and_without_drift_distribution",
                       )


    y = [[x[1] for x in qos_total] + [x[1] for x in qos_total_CD],
         [x[1] for x in qos_without_retrain] + [x[1] for x in qos_with_retrain],
         [x[1] for x in qos_without] + [x[1] for x in qos_with],
         [x[1] for x in qos_without_ll_only_SGD] + [x[1] for x in qos_with_ll_only_SGD],
         [x[1] for x in qos_without_ll_SGD_RF] + [x[1] for x in qos_with_ll_SGD_RF]
         ]
    plot_distributions([group_names] * len(y), y, names, colors, 
                       "Different scenarios", "Energy consumption (mC)", 0.06, 0.99, 
                       is_group = True,file_name = "ec_compare_with_and_without_drift_distribution",
                       )

    y = [[x[2] for x in qos_total] + [x[2] for x in qos_total_CD],
         [x[2] for x in qos_without_retrain] + [x[2] for x in qos_with_retrain],
         [x[2] for x in qos_without] + [x[2] for x in qos_with],
         [x[2] for x in qos_without_ll_only_SGD] + [x[2] for x in qos_with_ll_only_SGD],
         [x[2] for x in qos_without_ll_SGD_RF] + [x[2] for x in qos_with_ll_SGD_RF]
         ]
    plot_distributions([group_names] * len(y), y, names, colors, 
                       "Different scenarios", "Expected uitlity", 0.06, 0.3, 
                       is_group = True,file_name = "eu_compare_with_and_without_drift_distribution",
                       )

def snr_vis():
    data_with = []
    data_without = []
    stream_len = 1500
    for i in range (1, 1501):
        if(i <= 400):
            epsilon = 4.0/400.0
            epsilon1 = 2.0 + (i - 1) * epsilon
            epsilon2 = 2.0 + i * epsilon
        elif( (i > 400) and (i <= 900)):
            epsilon = 2.5/500.0
            epsilon2 = 6.0 - (i - 400 - 1) * epsilon
            epsilon1 = 6.0 - (i - 400) * epsilon
        elif((i > 900) and (i <= 1200)):
            epsilon = 2.0/300.0
            epsilon1 = 3.5 + (i - 900 - 1) * epsilon
            epsilon2 = 3.5 + (i - 900) * epsilon
        elif(i > 1200):
            epsilon = 2.5/300.0
            epsilon2 = 5.0 - (i - 1200 - 1) * epsilon
            epsilon1 = 5.0 - (i - 1200) * epsilon
            
        if(random.randint(1, 100) > 97):
            epsilon1 = 1.0
            epsilon2 = 4.0    

        data_without.append((epsilon2 - epsilon1) * random.random()  + epsilon1) 
        
        is_drifted = False
        if((i > 100) and (i <= 190)):
            is_drifted = True
            epsilon = 2.5/90.0
            epsilon2 = 8.0 + (i - 100 - 1) * epsilon
            epsilon1 = 8.0 + (i - 100) * epsilon

        elif((i > 190) and (i <= 350)):
            is_drifted = True
            epsilon = 2.0/160.0
            epsilon1 = 10.5 - (i - 190 - 1) * epsilon
            epsilon2 = 10.5 - (i - 190) * epsilon


        elif((i > 720) and (i <= 890)):
            is_drifted = True
            epsilon = 1.0/150.0
            epsilon2 = 9.0 - (i - 720 - 1) * epsilon
            epsilon1 = 9.0 - (i - 720) * epsilon

        elif((i > 890) and (i <= 1070)):
            is_drifted = True
            epsilon = 3.0/180.0
            epsilon1 = 10.0 + (i - 890 - 1) * epsilon
            epsilon2 = 10.0 + (i - 890) * epsilon

            

        if(is_drifted):
            if(random.randint(1, 100) > 97):
                epsilon1 = 8.0
                epsilon2 = 11.0
        data_with.append((epsilon2 - epsilon1) * random.random()  + epsilon1)   


    
    
    # data_with = []
    # data_without = []
    # for i in range(1500):
    #     if ((i > 300 and i <= 600) or (i > 900 and i <= 1200)):
    #         data_with.append(random.random() * 4 + 13)
        
    #     else:
    #         data_with.append(random.random() * 3 + 1)    
    #     data_without.append(random.random() * 3 + 1)
    
    plot_metrics([data_without, data_with],\
                 ["No concept drift", "Under concept drift"],\
                 stream_len, [[0,0], [100,100],[350,350],[720,720],[1070,1070], [1500, 1500]], [[-100,100],[-100,100],[-100,100],[-100,100],[-100,100],[-100,100]],\
                 "Network Inteference (dB)", "snrs", "No drift occured", y_range = [0, 40], add_vertical=True, line_colors=['rgb(175,214,17)','gray'])
    
    # stream_with = fetch_stream(stream_len, with_drift=True)    
    # stream_without = fetch_stream(stream_len, with_drift=False)
    
    # snrs_with = []
    # snrs_without = []
    
    # for i in range(stream_len):
    #     features, targets = stream_with.read_current_cycle()
    #     snrs_with.append(features[0][6]/10)

    #     features, targets = stream_without.read_current_cycle()
    #     snrs_without.append(features[0][6]*10)
        
    # plot_metrics([snrs_without, snrs_with],\
    #              ["SNR without drift", "SNR under drift"],\
    #              stream_len, [[300,300],[600,600],[900,900],[1200,1200]], [[-100,100],[-100,100],[-100,100],[-100,100]],\
    #              "SNR", "snrs", "No drift occured", y_range = [0, 40], add_vertical=True)

def plot_drifts():
    lines = []
    with open("./data/<path to task ids generated for lifelong scenarios in task_ids.txt>", "r+") as outfile:
        lines = outfile.readlines()
    
    line = lines[0]    
    task_ids = [int(x) for x in line.split(',') if len(x) > 0]
    task_ids_by_cycles = [[t] * 20 for t in task_ids] + [[task_ids[-1]] * 20 ]
    flatted_task_ids_by_cycles = list(itertools.chain.from_iterable(task_ids_by_cycles))    
    
    y_data = [flatted_task_ids_by_cycles[0]]
    x_data = [1]
    plots = []
    for i in range(1, 1500):
        if(flatted_task_ids_by_cycles[i] != flatted_task_ids_by_cycles[i-1]):
            plots.append(go.Scatter(x = x_data, y = y_data, line_width=5, line=dict(color=px.colors.qualitative.Dark24[flatted_task_ids_by_cycles[i-1]])))
            y_data = []
            x_data = []
        y_data.append(flatted_task_ids_by_cycles[i])
        x_data.append(i+1)
    
    
    plots.append(go.Scatter(x = x_data, y = y_data, line_width=5, line=dict(color=px.colors.qualitative.Dark24[flatted_task_ids_by_cycles[1499]])))
        
    fig = go.Figure(plots)
    
    cut_x_ranges = [[0,0], [100,100],[350,350],[720,720],[1070,1070], [1500, 1500]]
    for i in range(len(cut_x_ranges)-1):
        fig.add_vrect(
            x0=cut_x_ranges[i][0], x1=cut_x_ranges[i + 1][0],
            fillcolor="LightSalmon" if(i %2  == 0) else "LightSkyBlue", 
            opacity=0.2,
            line_width=0,
            layer="below", 
        )    
    
    
    fig.update_layout(
        yaxis = dict(
            title = "Task identifier"
        ),
        xaxis = dict(
            title = 'Adaptation cycle',
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend = False
    )           
    
    fig.show()
    fig.write_image("./figures/task_ids.pdf")

def plot_learning_time():
    lines = []
    with open("./data/<path to task ids generated for retraining scenarios in training_time.txt>", "r+") as outfile:
        lines = outfile.readlines()
    
    line = lines[0]    
    training_times_retrain = [float(x) for x in line.split(',') if len(x) > 0]
    
    with open("./data/<path to task ids generated for lifelong scenarios in training_time.txt>", "r+") as outfile:
        lines = outfile.readlines()
    
    line = lines[0]    
    training_times_incremental = [float(x) for x in line.split(',') if len(x) > 0]
    
    cycle_num = 1500
    
    plots = []
    plots.append(go.Scatter(x = list(range(2, cycle_num + 1)), y = np.cumsum(training_times_incremental), name = "SGD with LLL", 
                                line=dict(color="gray"), line_width=2))

    plots.append(go.Scatter(x = list(range(2, cycle_num + 1)), y = np.cumsum(training_times_retrain), name = "State-of-the-art",  marker_color = "lightseagreen",
                                line=dict(color="lightseagreen"), line_width=2))
    
    fig = go.Figure(data = plots)
    fig.update_layout(
        yaxis = dict(
            title = "Cumulative training time (s)"
        ),
        xaxis = dict(
            title = 'Adaptation cycle',
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend = True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.06
        )
    ) 
    fig.show()
    fig.write_image("./figures/training_time.pdf")
    
if __name__ == "__main__":
    run_feedbackloop()
    # run_ll_loop()
    # qos_visualization_multiple_goals_with_utility()
    # snr_vis()
    # learning_visualization()
    # task_detection()
    # plot_drifts()
    # plot_learning_time()