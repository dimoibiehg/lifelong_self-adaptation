class KonwledgeStore:
    def __init__(self):
        self.features = []
        self.selected_features = []
        self.selected_targets = {}
        self.enivornment_feautres = []
        # cycle ids for each task, 
        # here task id is the index of the list
        self.task_ids = [[]] 