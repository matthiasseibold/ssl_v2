import os
import librosa
import numpy as np
import pandas as pd
import soundfile
import matplotlib.pyplot as plt

plot = False
save = True

# init
root = "F:/SSL/experiment_3d/output/drilling/"
file = "1_022_Movie2D_heatmap.wav"

count = 1
print("Processing file: " + file)

# read labels
filename, file_extension = os.path.splitext(file)
labels = pd.read_csv("../../../labels/drilling/"+ filename + ".csv", header=None).to_numpy()

# create subfolder for specimen
save_path = "F:/datasets/ssl_v2/long_file_drilling_" + filename
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_path + "/" + filename, exist_ok=True)


# read sample
wav2, sr = librosa.load(root + "/" + file,
                        sr=None, mono=True)

# Define window and hop (in seconds)
window_duration = 0.15  # 150 ms
hop_duration = 0.02  # 20 ms overlap

# Convert to samples
window_size = int(window_duration * sr)
hop_size = int(hop_duration * sr)

# create figure
if plot:
    plt.figure()
    plt.tight_layout()


# Sliding window loop
for start in range(0, len(wav2) - window_size + 1, hop_size):

    print("Window #" + str(count))

    end = start + window_size
    window = wav2[start:end]


    start_s = start / sr
    end_s = end / sr

    # check if we have a peak in the spectrogram (and allow for some slipping)
    condition = np.any((start_s >= labels[:, 0]) & (end_s <= labels[:, 1]))

    # write the file
    soundfile.write(save_path + "/" + filename + "/" + str(count).zfill(5) + file_extension, window, 16000)

    if save:
        if condition:
            if count == 1:
                label = np.ones(1)
            else:
                label = np.append(label, 1)
        else:
            if count == 1:
                label = 0 * np.ones(1)
            else:
                label = np.append(label, 0)

     # Plot
    if plot:
        S = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=128, fmax=8000, hop_length=256)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
        if condition:
           plt.title('drilling')
        else:
            plt.title('No drilling')
        plt.draw()
        plt.pause(0.01)

    count += 1

os.makedirs("../data_ast/drilling_long_" + filename, exist_ok=True)
np.save("../data_ast/drilling_long_" + filename + "/test_y.npy", label)

