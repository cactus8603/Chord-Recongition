
import matplotlib.pyplot as plt
import librosa, librosa.display

import numpy as np
import json
"""
audio_file = "audio/1.mp3"
y, sr = librosa.load(audio_file, sr=22050)
tmp = librosa.cqt(y, sr=22050, n_bins=84, bins_per_octave=12, hop_length=512)
"""

# Create templates for major, minor, and no-chord qualities
maj_template = np.array([1,0,0, 0,1,0, 0,1,0, 0,0,0])
min_template = np.array([1,0,0, 1,0,0, 0,1,0, 0,0,0])
N_template   = np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1.]) / 4.
# Generate the weighting matrix that maps chroma to labels
weights = np.zeros((25, 12), dtype=float)

labels = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj',
          'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
          'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min',
          'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
          'N']
"""
labels = ['C:maj', 'C:min', 'C#:maj', 'C#:min', 'D:maj', 'D:min', 'D#maj', 'D#:min', 'E:maj', 'E:min', 'F:maj', 'F:min', 'F#:maj',
             'F#:min', 'G:maj', 'G:min', 'G#:maj', 'G#:min', 'A:maj', 'A:min', 'A#:maj', 'A#:min', 'B:maj', 'B:min', 'N']
"""
root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']

for c in range(12):
    weights[c, :] = np.roll(maj_template, c) # c:maj
    weights[c + 12, :] = np.roll(min_template, c)  # c:min
weights[-1] = N_template  # the last row is the no-chord class
print(weights)
# Make a self-loop transition matrix over 25 states
trans = librosa.sequence.transition_loop(25, 0.95)

with open('./audio/feature.json', 'r') as f:
    json_data = json.load(f)
f.close()

# chroma = np.array(json_data['chroma_stft'])
chroma = np.array(json_data['chroma_cqt'])
np.savetxt("chroma_cqt.txt", chroma , delimiter=",")
print(chroma[0][1])
# chroma = np.array(json_data['chroma_cens'])


# Map chroma (observations) to class (state) likelihoods
probs = np.exp(weights.dot(chroma))  # P[class | chroma] ~= exp(template' chroma)
probs /= probs.sum(axis=0, keepdims=True)  # probabilities must sum to 1 in each column
# Compute independent frame-wise estimates
chords_ind = np.argmax(probs, axis=0)
# And viterbi estimates
chords_vit = librosa.sequence.viterbi_discriminative(probs, trans)
# Plot the features and prediction map
# print(chords_ind)
"""
with open ("chords_int.txt", "w") as f:
    for row in chords_ind:
        np.savetxt(f,row) 
    f.close()
"""
# np.savetxt("chords_int_stft.txt", chords_ind , delimiter=",")
# np.savetxt("chords_vit_stft.txt", chords_vit , delimiter=",")

### for chroma_cens ###
"""
plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
plt.colorbar()
plt.subplot(2,1,2)
librosa.display.specshow(weights, x_axis='chroma')
plt.yticks(np.arange(25) + 0.5, labels)
plt.ylabel('Chord')
plt.colorbar()
plt.tight_layout()

# And plot the results
plt.figure(figsize=(10, 4))
librosa.display.specshow(probs, x_axis='time', cmap='gray')
plt.colorbar()
times = librosa.frames_to_time(np.arange(len(chords_vit)))
plt.scatter(times, chords_ind + 0.75, color='lime', alpha=0.5, marker='+', s=15, label='Independent')
plt.scatter(times, chords_vit + 0.25, color='deeppink', alpha=0.5, marker='o', s=15, label='Viterbi')
plt.yticks(0.5 + np.unique(chords_vit), [labels[i] for i in np.unique(chords_vit)], va='center')
plt.legend()
plt.tight_layout()
plt.show()
"""
### Finish ###


fig, ax = plt.subplots()
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
ax.set(title='Chromagram demonstration')
fig.colorbar(img, ax=ax)

# plt.show()

fig, ax = plt.subplots(nrows=2)
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', ax=ax[0])
librosa.display.specshow(weights, x_axis='chroma', ax=ax[1])
ax[1].set(yticks=np.arange(25) + 0.5, yticklabels=labels, ylabel='Chord')
# And plot the results
fig, ax = plt.subplots()
librosa.display.specshow(probs, x_axis='time', cmap='gray', ax=ax)
times = librosa.times_like(chords_vit)


# print("times = ", times)
np.savetxt("times_stft.txt", times , delimiter=",")

ax.scatter(times, chords_ind+0.75 , color='lime', alpha=0.5, marker='+',
           s=15, label='Independent')
ax.scatter(times, chords_vit+0.25 , color='deeppink', alpha=0.5, marker='o',
           s=15, label='Viterbi')
ax.set(yticks=0.5 + np.unique(chords_vit),
       yticklabels=[labels[i] for i in np.unique(chords_vit)])
ax.legend(loc='upper right')

# plt.show()

chroma_med = librosa.decompose.nn_filter(chroma, aggregate=np.median, metric='cosine')
rec = librosa.segment.recurrence_matrix(chroma, mode='affinity', metric='cosine', sparse=True)
chroma_nlm = librosa.decompose.nn_filter(chroma, rec=rec, aggregate=np.average)

np.savetxt("chroma_cqt_med.txt", chroma_med , delimiter=",")
np.savetxt("chroma_cqt_nlm.txt", chroma_nlm , delimiter=",")


fig, ax = plt.subplots(nrows=5, sharex=True, sharey=True, figsize=(10, 10))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[0])
ax[0].set(title='Unfiltered')
ax[0].label_outer()
librosa.display.specshow(chroma_med, y_axis='chroma', x_axis='time', ax=ax[1])
ax[1].set(title='Median-filtered')
ax[1].label_outer()
imgc = librosa.display.specshow(chroma_nlm, y_axis='chroma', x_axis='time', ax=ax[2])
ax[2].set(title='Non-local means')
ax[2].label_outer()
imgr1 = librosa.display.specshow(chroma - chroma_med,
                         y_axis='chroma', x_axis='time', ax=ax[3])
ax[3].set(title='Original - median')
ax[3].label_outer()
imgr2 = librosa.display.specshow(chroma - chroma_nlm,
                         y_axis='chroma', x_axis='time', ax=ax[4])
ax[4].label_outer()
ax[4].set(title='Original - NLM')
fig.colorbar(imgc, ax=ax[:3])
fig.colorbar(imgr1, ax=[ax[3]])
fig.colorbar(imgr2, ax=[ax[4]])

plt.show()

