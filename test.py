import numpy as np
import json
import librosa
from total_chord import *

def getData(feature_path):
    with open(feature_path, 'r') as f:
        json_data = json.load(f)

    # chroma = np.array(json_data['chroma_stft'])
    chroma = np.array(json_data['chroma_cqt'])

    # chroma = librosa.decompose.nn_filter(chroma, aggregate=np.median, metric='cosine')
    # chroma = librosa.decompose.nn_filter(chroma, aggregate=np.average)
    length = len(chroma[0])
    times = librosa.times_like(length)
    # np.savetxt("data/times.txt", times , delimiter=",")
    
    tmp = np.array([[chroma[0]],
                    [chroma[1]],
                    [chroma[2]],
                    [chroma[3]],
                    [chroma[4]],
                    [chroma[5]],
                    [chroma[6]],
                    [chroma[7]],
                    [chroma[8]],
                    [chroma[9]],
                    [chroma[10]],
                    [chroma[11]],
                    ])
    # data = tmp.T
    # data = np.reshape(data, (8673, 12))
    # sort = np.argsort(tmp, axis=0, kind='heapsort')
    # sort = np.argmax(data, axis=1)
    # print(np.shape(sort))
    sort = np.argmax(tmp, axis=0)
    sort = np.reshape(sort, (length,))
    # print(sort)
    # print(np.shape(sort))
    return chroma, times

def chord_reg(chroma, sort, order, loss_rate):
    chord_dict = {
        0:'C',
        1:'C#',
        2:'D',
        3:'D#',
        4:'E',
        5:'F',
        6:'F#',
        7:'G',
        8:'G#',
        9:'A',
        10:'A#',
        11:'B',

    }

    i = 0
    # order = 5905
    tmp_sum = 0
    for i in range(12):
        tmp_sum = tmp_sum + chroma[i][order]
        # print("order=", i, "sord =  ", sort[order]," value = ", chroma[i][order])

    tmp = chroma[(sort[order]+3)%12][order]
    tmp1 = chroma[(sort[order]+4)%12][order]
    tmp_average = tmp_sum/12
    # N_or_not = (tmp_sum - chroma[sort[order]][order]) / 9
    loss = abs(tmp_average - tmp) / tmp_average
    loss1 = abs(tmp_average - tmp1) / tmp_average
    # loss = abs(N_or_not - tmp) / N_or_not
    # loss1 = abs(N_or_not - tmp1) / N_or_not
   
    if(tmp > tmp1): 
        tmp_sum = tmp_sum - chroma[sort[order]][order] - chroma[(sort[order]+3)%12][order] - chroma[(sort[order]+7)%12][order]
        # print("n+10", chroma[(sort[order]+10)%12][order])
        # print("average = ",tmp_sum/9)
        if(loss < loss_rate):
            # print("N")
            return str("N")
        else:
            # print(chord_dict[sort[order]])
            return str(chord_dict[sort[order]]+":min")
    else: 
        tmp_sum = tmp_sum - chroma[sort[order]][order] - chroma[(sort[order]+4)%12][order] - chroma[(sort[order]+7)%12][order]
        # print("n+10", chroma[(sort[order]+10)%12][order])
        # print("n+11", chroma[(sort[order]+11)%12][order])
        # print("average = ",tmp_sum/9)

        if(loss1 < loss_rate):
            # print("N")
            return str("N")
        else:
            # print(chord_dict[sort[order]])
            return str(chord_dict[sort[order]]+":maj")

def getTrain_data(path):
    for i in range(8,11):
        feature_path = ".//data//"+str(i)+"//feature.json"
        ans_path = ".//data//"+str(i)+"//ground_truth.txt"
        chroma, times = getData(feature_path)
        ans = getAns(ans_path)

        ans_per_frame = np.array([])
        ans_index = 0
        print(float(ans['end_time'][len(ans['chord'])-1]))

        frame = 0
        
        
        while(frame < len(chroma[0])-1):
            
            if(times[frame] < float(ans['end_time'][ans_index])):
                ans_per_frame = np.append(ans_per_frame, ans['chord'][ans_index])
                frame = frame + 1
            else :
                ans_index = ans_index + 1
                if(ans_index == len(ans['chord'])):break

            
            
        print(i," finish")
        print(len(ans_per_frame))
        print(ans_per_frame[5000])

        np.save("data//train_data//" + str(i), ans_per_frame)


if __name__ == "__main__" :
    """
    for i in range(8,11):
        feature_path = ".//data//"+str(i)+"//feature.json"
        ans_path = ".//data//"+str(i)+"//ground_truth.txt"
        chroma, times = getData(feature_path)
        ans = getAns(ans_path)

        ans_per_frame = np.array([])
        ans_index = 0
        print(float(ans['end_time'][len(ans['chord'])-1]))

        frame = 0
        
        
        while(frame < len(chroma[0])-1):
            
            if(times[frame] < float(ans['end_time'][ans_index])):
                ans_per_frame = np.append(ans_per_frame, ans['chord'][ans_index])
                frame = frame + 1
            else :
                ans_index = ans_index + 1
                if(ans_index == len(ans['chord'])):break

            
            
        print(i," finish")
        print(len(ans_per_frame))
        print(ans_per_frame[5000])

        np.save("data//train_data//" + str(i), ans_per_frame)
    """

    """
    for i in range(1, 11):
        path = "data//train_data//" + str(i) + ".npy"
        x = np.load(path)
        print(i)
        print(len(x))
        print(x[len(x)-1])
        # print(chord_reg(chroma, sort, 2746, 0.4))
        
        # dict_chord = getDict_Chord()
        # print(dict_chord)
    """

