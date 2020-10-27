import json
import numpy as np
with open('feature.json', 'r') as f:
    # feature, feature_per_second, song_length_second = json.loads(f)
    data = json.load(f)

# print(data['chroma_cqt'])
print(np.shape(data))