class KonwledgeStore:
    def __init__(self):
        # this is not related to triplets in the paper, 
        # triplets not directly useful for simulation here.
        self.data_pair = []  
        self.task_ids = [[]] 
        self.cache_current_flow_data_up_to_reach_operator_cycle = [] # for task-based knowledge miner
        self.cache_previous_operator_fed_labeled_data = []