from typing import overload
from os import path
import json
from common.goal import *
class Stream:

    def __init__(self, stream_len, target_types = [TargetType.LATENCY, TargetType.PACKETLOSS, TargetType.ENERGY_CONSUMPTION],
                                    target_names = ["latency", "packetloss", "energyconsumption"], stream_addr = "./data/total", stream_file_base_name = "dataset_with_all_features"):
        self.__base_addr = stream_addr
        self.cycles_num = stream_len
        self.__file_base_name = stream_file_base_name
        self.__target_names = target_names
        self.__target_types = target_types
        self.current_cyle = 1

    def read_current_cycle(self):
        if(self.current_cyle > self.cycles_num):
            exit("end of stream")
        current_stream = self.__load_raw_features_qualities(self.current_cyle)
        self.current_cyle += 1
        return current_stream
    
    def read(self, cycle_num: int):
        specified_stream = self.__load_raw_features_qualities(cycle_num)
        return specified_stream


    def __flatten(self, l):
        return [e for sublist in l for e in sublist]


    def __load_raw_data(self, cycle_num:int):
        r = []
        with open(path.join(self.__base_addr, f'{self.__file_base_name}{cycle_num}.json'), 'r') as f:
            r.append(json.load(f))

        return r


    def __load_raw_data_split(self, cycle_num:int, target_name:str):
        data = self.__load_raw_data(cycle_num)
        return 	[c['features'] for c in data], [c[target_name] for c in data]



    def __load_raw_features_qualities(self, cycle_num:int):
        
        target_vals = {}
        for i in range(len(self.__target_types)):
            features, target = self.__load_raw_data_split(cycle_num, self.__target_names[i])
            target_vals[self.__target_types[i]] = self.__flatten(target)
        return self.__flatten(features), target_vals