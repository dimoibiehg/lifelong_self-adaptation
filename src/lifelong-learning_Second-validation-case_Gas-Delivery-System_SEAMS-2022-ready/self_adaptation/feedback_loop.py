from common.stream import Stream
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
from lifelong_learner.lifelong_learning_loop import *


class FeedbackLoop:
    def __init__(self, stream : Stream, lifelong_loop, operator_support = False):
        self.stream = stream
        self.X = []
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.calssifier_model = OneVsOneClassifier(SVC(random_state=0,kernel='rbf'))
        self.operator_cycle = 100
        self.accs_baseline = []
        self.y_pred = []
        self.operator_support = operator_support
        self.lifelong_loop = lifelong_loop
        # self.offline_phase()
        
        
    def offline_phase(self):
        dataX = []
        dataY = []
        for  i in range(self.operator_cycle):
            dataX.append(self.stream.read_current_cycle())
            dataY.append(self.stream.Y_features[i])
        
        if(self.lifelong_loop is not None):
            self.lifelong_loop.knowledge.data_pair.extend(list(zip(dataX, dataY)))
            self.lifelong_loop.knowledge.cache_current_flow_data_up_to_reach_operator_cycle.extend(list(zip(dataX, dataY)))
        
        scaled_features = self.scaler.fit_transform(dataX)
        self.calssifier_model.fit(scaled_features, dataY)
    def monitor(self):
        if(self.operator_support):
            self.X.append(self.stream.read_current_cycle())    
        else:
            self.X = [self.stream.read_current_cycle()]
        
        if(self.lifelong_loop is not None):
            self.lifelong_loop.knowledge.data_pair.append((self.X[-1], -1))
        self.classifier()
        
    def classifier(self):
        y_pred = self.calssifier_model.predict(self.scaler.transform([self.X[-1]]))
        
        #compute metrics (for validation) / out of adaptation scope
        acc = accuracy_score([self.stream.Y_features[self.stream.current_cyle-2]], y_pred)
        self.accs_baseline.append(acc)
        self.y_pred.append(y_pred)
        
        if(self.operator_support):    
            if(len(self.y_pred) >= self.operator_cycle):
                # call operator to correct labels
                corrected_y_pred = self.stream.Y_features[(self.stream.current_cyle-1-self.operator_cycle):(self.stream.current_cyle-1)]
                scaled_features = self.scaler.fit_transform(self.X)
                self.calssifier_model.fit(scaled_features, corrected_y_pred)
                if self.lifelong_loop is not None:
                    self.lifelong_loop.knowledge.data_pair[-self.operator_cycle:] = list(zip(self.X, corrected_y_pred))
                    self.lifelong_loop.knowledge.cache_previous_operator_fed_labeled_data = list(zip(self.X, corrected_y_pred))
                self.y_pred = []
                self.X = []
                
        else:
            self.y_pred =[y_pred]
        self.plan_and_execute()
    def plan_and_execute(self):
        ## open the corresponding MFC to route the self.y_pred[-1] gas
        pass 