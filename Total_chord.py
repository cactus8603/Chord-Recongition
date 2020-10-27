import numpy as np
# import json
# import pandas as pd

def getDict_Chord():
    dict_chord = {}

    data = np.array(['N'])
    for i in range(1,201):
        path = "D:\\others\\桌面\\project\\ai-cup\\CE200\\CE200\\" + str(i) + "\\ground_truth.txt"
        # path = ""
        with open(path, "r") as f:
            for lines in f.readlines():
                lines = lines.rstrip('\n').rstrip(' ')
                lines = lines.replace("\t"," ")
                pos = lines.find(':')
                if(pos>0):
                    if(lines[(pos-2)] != ' '): index = pos-2
                    else: index = pos - 1
                    if lines[index:] not in data:
                        data = np.append(data, lines[index:])
        

    data = np.sort(data)
    
    for i in range(len(data)):
        dict_chord[i] = data[i]
    
    # pd.DataFrame(data).to_json(".\\data\\total_chord.json", orient='split')

    return dict_chord


def getAns(path):
    
    # path = "D:\\others\\桌面\\project\\ai-cup\\CE200\\CE200\\" + str(i) + "\\ground_truth.txt"
    # path = ".\\data\\1\\ground_truth.txt"

    data = {
        'start_time':[],
        'end_time':[],
        'chord':[],
    }

    with open(path, "r") as f:
        for lines in f.readlines():
    
            lines = lines.rstrip('\n').rstrip(' ')
            lines = lines.replace("\t",",")

            pos_end = lines.find(',')            
            pos_chord = lines.find(',', pos_end+1)

            data['start_time'] = np.append(data['start_time'], lines[:pos_end])
            data['end_time'] = np.append(data['end_time'], lines[pos_end+1:pos_chord])
            data['chord'] = np.append(data['chord'], lines[pos_chord+1:])

    return data
    


if __name__ == "__main__":
    # print(getDict_Chord())
  