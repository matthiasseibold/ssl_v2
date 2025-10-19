import os
import librosa
import numpy as np
import pandas as pd
import soundfile
import matplotlib.pyplot as plt

plot = False
save = True

# init
root_dir = "F:/SSL/experiment_3d/output/sawing"
save_path_peak = "F:/datasets/ssl_v2/sawing_ast"
save_path_nopeak = "F:/datasets/ssl_v2/nosawing_ast"
os.makedirs(save_path_peak, exist_ok=True)
os.makedirs(save_path_nopeak, exist_ok=True)

files_raw = os.listdir(root_dir)
wav_files = [f for f in files_raw if ".wav" in f]

for file in wav_files:

    count = 1
    print("Processing file: " + file)

    # read labels
    filename, file_extension = os.path.splitext(file)
    labels = pd.read_csv("../../../labels/sawing/"+ filename + ".csv", header=None).to_numpy()

    # create subfolder for specimen
    os.makedirs(save_path_peak + "/" + filename, exist_ok=True)
    os.makedirs(save_path_nopeak + "/" + filename, exist_ok=True)

    # read sample
    wav2, sr = librosa.load(root_dir + "/" + file,
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

        # print("Window #" + str(count))

        end = start + window_size
        window = wav2[start:end]
        count += 1

        start_s = start / sr
        end_s = end / sr

        # check if we have a sawing in the spectrogram
        condition = end_s > labels[0,0] and start_s < labels[0,1]

        if save:
            if condition:
                soundfile.write(save_path_peak + "/" + filename + "/" + str(count).zfill(5) + file_extension, window, 16000)
            else:
                soundfile.write(save_path_nopeak + "/" + filename + "/" + str(count).zfill(5) + file_extension, window, 16000)

        # Plot
        if plot:
            S = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=128, fmax=8000, hop_length=256)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
            if condition:
                plt.title('Sawing')
            else:
                plt.title('No Sawing')
            plt.draw()
            plt.pause(0.01)


