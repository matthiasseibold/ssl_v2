import os
import librosa.display
import soundfile

target_path = "../long_file_windows/"
file = "F:/datasets/New_SwallowSet/Raw/Healthy/217/long_audio/ZOOM0009_Tr1.WAV"

if not os.path.exists(target_path):
    os.makedirs(target_path)

if "Healthy" in file:
    path_extension = "Healthy/"
elif "Idle" in file:
    path_extension = "Idle/"
else:
    path_extension = "Zenker/"

target_path = target_path + path_extension
if not os.path.exists(target_path):
    os.makedirs(target_path)

# we need to create another subfolder to have the same structure as the dataset and being able to
# reuse the PrepareTrainingData_AST file
target_path = target_path + "subfolder/"
if not os.path.exists(target_path):
    os.makedirs(target_path)

# load file
y, sr = librosa.load(file)
file_length = len(y)
win_length = sr
number_of_steps = (file_length / win_length) * 2

for i in range(int(number_of_steps)-1):
    window_start = 0 + i * win_length / 2
    window_end = win_length + i * win_length / 2
    window_y = y[int(window_start):int(window_end)]
    soundfile.write(target_path + "window_" + str(i) + ".wav", window_y, sr)