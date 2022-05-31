import random 


# Data batches fecthed from this link: http://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations
class Stream:
    def __init__(self, stream_len):
        self.cycles_num = stream_len
        self.current_cyle = 1
        self.X_features = []
        self.Y_features = []
        X, Y = self.read_data()
        
        for i in range(len(X)):
            l = list(enumerate(X[i]))
            random.shuffle(l)
            indices, shuffled_x = zip(*l)
            shuffled_y =  [Y[i][indices[j]] for j in range(len(Y[i]))]
            self.X_features.extend(shuffled_x)
            self.Y_features.extend(shuffled_y)
    def read_current_cycle(self):
        if(self.current_cyle > self.cycles_num):
            exit("end of stream")
        current_stream = self.X_features[self.current_cyle-1]
        self.current_cyle += 1
        return current_stream
    
    def read_data(self):
        X = []
        Y = []
        for i in range(1, 11):
            with open(f"./data/batch{i}.dat", "r") as gas_batch:
                X.append([])
                Y.append([])
                line = gas_batch.read()
                splitted_data = [[y for y in x.split(' ') if len(y) > 0] for x in line.split('\n') if len(x) > 0]
                # print(splitted_data)
                
                for record in splitted_data:
                    X[i-1].append([float(x.split(':')[1]) for x in record[1:]])
                    Y[i-1].append(int(record[0].split(";")[0]))
                        
                # print(np.where(np.array(Y[i-1]) == 3)[0].shape)
                # print(np.array(Y[i-1]).shape)
        return X, Y    
        
        