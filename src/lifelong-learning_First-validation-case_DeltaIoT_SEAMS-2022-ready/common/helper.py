import math 
import sys, os

def highestPowerof2(n):
    p = int(math.log(n, 2))
    return int(pow(2, p))

# At the moment, there is a bug in Keras Tuner that you cannot control the verbosity.
# Hence, two following functions can be helful in this kind of case.
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__