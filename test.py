import numpy as np
import json
import librosa

def getData(feature_path):
    with open(feature_path, 'r') as f:
        json_data = json.load(f)
    f.close()

    # chroma = np.array(json_data['chroma_stft'])
    chroma = np.array(json_data['chroma_cqt'])

    # chroma = librosa.decompose.nn_filter(chroma, aggregate=np.median, metric='cosine')
    # chroma = librosa.decompose.nn_filter(chroma, aggregate=np.average)

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
    sort = np.reshape(sort, (8673,))
    # print(sort)
    # print(np.shape(sort))
    return chroma, sort

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
    N_or_not = (tmp_sum - chroma[sort[order]][order]) / 9
    # - chroma[(sort[order]-1)%12][order] - chroma[(sort[order]+1)%12][order]
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


if __name__ == "__main__" :
    feature_path = "./audio/feature.json"
    chroma, sort = getData(feature_path)
    for i in range (8480,8673):
        print(i, chord_reg(chroma, sort, i, 0))
        # print(i, chord_reg(chroma, sort, i, 0.05), chroma[(sort[i]+3)%12][i], chroma[(sort[i]+4)%12][i])
    
    # print(chord_reg(chroma, sort, 2746, 0.4))



