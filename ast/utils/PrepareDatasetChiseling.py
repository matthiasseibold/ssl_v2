import os
import librosa
import numpy as np
import pandas as pd
import soundfile
import matplotlib.pyplot as plt

# init
root_dir = "F:/SSL/experiment_3d/output/chiseling"
save_path_peak = "F:/SSL/experiment_3d/output/chiseling/peak"
save_path_nopeak = "F:/SSL/experiment_3d/output/chiseling/nopeak"
os.makedirs(save_path_peak, exist_ok=True)
os.makedirs(save_path_nopeak, exist_ok=True)

files_raw = os.listdir(root_dir)
wav_files = [f for f in files_raw if ".wav" in f]

for file in wav_files:

    count = 1
    print("Processing file: " + file)

    # read labels
    filename, file_extension = os.path.splitext(file)
    labels = pd.read_csv("../../Labels/chiseling/"+ filename + ".csv", header=None).to_numpy()

    # read sample
    wav2, sr = librosa.load(root_dir + "/" + file,
                            sr=None, mono=True)

    # Define window and hop (in seconds)
    window_duration = 0.2  # 200 ms
    hop_duration = 0.05  # 50 ms overlap

    # Convert to samples
    window_size = int(window_duration * sr)
    hop_size = int(hop_duration * sr)

    # create figure
    plt.figure(figsize=(10, 4))
    plt.title('Mel Spectrogram')
    plt.tight_layout()

    # Sliding window loop
    for start in range(0, len(wav2) - window_size + 1, hop_size):
        end = start + window_size
        window = wav2[start:end]
        count += 1

        # Compute Mel spectrogram
        S = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=128, fmax=8000)

        # Convert to decibels (log scale)
        S_dB = librosa.power_to_db(S, ref=np.max)

        start_s = start / sr
        end_s = end / sr

        # check if we have a peak in the spectrum
        condition = np.any((labels >= start_s) & (labels <= end_s))

        # Plot
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
        if condition:
            plt.title('Peak')
        else:
            plt.title('No peak')
        plt.show()


