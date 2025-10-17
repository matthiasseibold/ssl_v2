import numpy as np
import matplotlib.pyplot as plt
import librosa

audio_file = "F:/SSL/experiment_3d/output/chiseling/1_011_Movie2D_heatmap.wav"
waveform, sr = librosa.load(audio_file, sr=None)

# Example ground truth and predicted event times (in the same time units as t)
ground_truth_events = np.array([0.5, 1.5, 4.5])
predicted_events = np.array([1.0, 2.0, 5.0])

# Plot the waveform
librosa.display.waveshow(waveform, sr=sr, alpha=0.6)

# Parameters for arrows
arrow_length = 0.3
arrow_width = 0.02
y_top = max(waveform) + 0.2  # position arrows just above waveform top
y_bottom = min(waveform) - 0.5  # position arrows below waveform bottom

# Plot red arrows for ground truth events on top
for gt in ground_truth_events:
    plt.arrow(gt, y_top, 0, -arrow_length, head_width=0.2, head_length=0.1, fc='red', ec='red')

# Plot green arrows for predicted events on bottom
for pred in predicted_events:
    plt.arrow(pred, y_bottom, 0, arrow_length, head_width=0.2, head_length=0.1, fc='green', ec='green')

# Add labels and adjust plot
plt.ylim(y_bottom - 0.3, y_top + 0.3)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Waveform with Ground Truth (red) and Predicted (green) Events')
# plt.legend(['Waveform', 'Ground Truth Events', 'Predicted Events'])
plt.show()