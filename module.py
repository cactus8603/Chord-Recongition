import tensorflow as tf
import numpy as np 
from total_chord import *
from test import *

if __name__ == "__main__":

    # prepare train and test data
    # ans: start, end, chord
    # for ML: feature, sort, times
    
    dict_total = getDict_Chord()
    # print(dict_total)
    path = ".\\data\\1\\ground_truth.txt"
    ans = getAns(path)
    print(ans)
    # path = ""
    # ans = getAns(path)