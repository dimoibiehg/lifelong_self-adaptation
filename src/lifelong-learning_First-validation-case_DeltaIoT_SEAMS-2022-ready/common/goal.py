from enum import Enum
from strenum import StrEnum
from abc import ABC
import math 
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from random import randint

class GoalType(Enum):
    THRESHOLD = 1
    OPTIMIZATION = 2

class CompareType(Enum):
    GREATER = 1
    LESS = 2

class OptimizationType(Enum):
    MAX = 1
    MIN = 2
class TargetType(Enum):
    def __str__(self):
        return self.name
    LATENCY = 1
    PACKETLOSS = 2
    ENERGY_CONSUMPTION = 3


class TargetName(StrEnum):
    LATENCY = "latency"
    PACKETLOSS = "packetloss"
    ENERGY_CONSUMPTION = "energyconsumption"


class TargetRange(Enum):
    LATENCY = [0,100]
    PACKETLOSS = [0,100]
    ENERGY_CONSUMPTION = [0, math.inf]

class Goal(ABC):
    def __init__(self, goal_type, target_type):
        self.type = goal_type
        self.target_type = target_type

class ThrehsholdGoal(Goal):
    def __init__(self, target_type, compare_type, value):
        self.compare_type = compare_type
        self.value = value
        super().__init__(GoalType.THRESHOLD, target_type)

class OptimizationGoal(Goal):
    def __init__(self, target_type, optimization_type):
        self.optimization_type = optimization_type
        super().__init__(GoalType.OPTIMIZATION, target_type)


def utility_fucntion(target_type, value):
    if(target_type == TargetType.PACKETLOSS):
        return -0.01 * value + 1
    elif(target_type == TargetType.ENERGY_CONSUMPTION):
        if((value <= 13.0) and (value >= 0.0)):
            return 1.0
        elif((value > 13.0) and (value <= 13.4)):
            return 0.25
        elif(value > 13.4):
            return 0
        else:
            raise Exception("Not in the range of energy consumtpion")

    else:
        raise Exception("TergetType other than packet loss and energy consumtpion has not been implemented.")
    
def weight_for_targte_type(target_type):
    if(target_type == TargetType.PACKETLOSS):
        return 0.8
    elif(target_type == TargetType.ENERGY_CONSUMPTION):
        return 0.2
    else:
        raise Exception("TergetType other than packet loss and energy consumtpion has not been implemented.") 