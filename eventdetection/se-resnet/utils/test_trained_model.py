import os
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import resnet

plot = True

# init
root = "F:/SSL/experiment_3d/output/chiseling/"
file = "1_016_Movie2D_heatmap.wav"

model = resnet.ResNet18(input_shape=(128, 26, 1), classes=2, use_se=True)
model.load_weights("../checkpoints/trained_model.hdf5")

count = 1
print("Processing file: " + file)

# read labels
filename, file_extension = os.path.splitext(file)
labels = pd.read_csv("../../labels/chiseling/"+ filename + ".csv", header=None).to_numpy()

# read sample
wav2, sr = librosa.load(root + file,
                        sr=None, mono=True)

# Define window and hop (in seconds)
window_duration = 0.15  # 150 ms
hop_duration = 0.05  # 50 ms overlap

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
    count += 1

    start_s = start / sr
    end_s = end / sr

    # mel spec
    S = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=128, fmax=8000, hop_length=256)
    S_dB_raw = librosa.power_to_db(S, ref=np.max)

    # normalize it
    mean = -25.694532
    std = 8.431269
    S_dB = (S_dB_raw - mean) / std
    S_dB = np.reshape(S_dB, (1, 128, 26, 1))

    # model prediction
    prediction = model.predict(S_dB)
    prediction_argmax = prediction.argmax(axis=1)
    pred_str = "No Peak"
    if prediction_argmax == True:
        pred_str = "Peak"

    # check if we have a peak in the spectrogram (and allow for some slipping)
    condition = np.any((labels >= start_s - 0.05) & (labels <= end_s))

    # Plot
    if plot:
        librosa.display.specshow(S_dB_raw, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
        if condition:
            plt.title('Prediction: ' + pred_str + ' --- Groundtruth: Peak')
        else:
            plt.title('Prediction: ' + pred_str + ' --- Groundtruth: No Peak')
        plt.draw()
        plt.pause(0.01)


