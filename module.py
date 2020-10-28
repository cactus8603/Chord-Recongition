import tensorflow as tf
import numpy as np 
from total_chord import *
from test import *

if __name__ == "__main__":

    # prepare train and test data
    # ans: start, end, chord
    # for ML: feature, sort, times
    
    dict_total = getDict_Chord()
    print(len(dict_total))
    
    # data['start_time'], data['end_time'], data['chord']
    path = ".\\data\\1\\ground_truth.txt"
    ans = getAns(path)

    path = ".\\data\\1\\feature,json"
    chroma, times = getData(path)


    # path = ""
    # ans = getAns(path)