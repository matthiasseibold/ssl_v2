import os
import librosa
import numpy as np
import soundfile
from pydub import AudioSegment

# init
count = 0
root_dir = "F:/datasets/New_SwallowSet/Raw"
save_path = "F:/datasets/New_SwallowSet/Augmented/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
classes = os.listdir(root_dir)

# iterate through the dataset
for cl in classes:

    if not os.path.exists(save_path + cl + "/"):
        os.makedirs(save_path + cl + "/")

    specimens = os.listdir(root_dir + "/" + cl)

    for specimen in specimens:

        if not os.path.exists(save_path + cl + "/" + specimen + "/"):
            os.makedirs(save_path + cl + "/" + specimen + "/")

        subfolder = os.listdir(root_dir + "/" + cl + "/" + specimen)
        subfolder = [f for f in subfolder if not ".csv" in f]

        files = os.listdir(root_dir + "/" + cl + "/" + specimen + "/" + subfolder[0])
        wav_files = [f for f in files if ".wav" in f]

        for file in wav_files:

            print("Processing file: " + file)

            # split file extension
            filename, file_extension = os.path.splitext(file)

            # read sample
            # wav = AudioSegment.from_wav(root_dir + "/" + cl + "/" + specimen + "/" + file)
            wav2, sr = librosa.load(root_dir + "/" + cl + "/" + specimen + "/" + subfolder[0] + "/" + file,
                                    sr=None, mono=True)

            # copy original sample
            soundfile.write(save_path + "/" + cl + "/" + specimen + "/" + file, wav2, sr)

            # change volume gain
            # wav_louder = wav + 5
            # wav_lower = wav - 5
            # # wav.export(save_path + f + "/" + filename + "_raw" + file_extension, format='wav')
            # wav_louder.export(save_path + "/" + cl + "/" + specimen + "/" + filename + "_louder"
            #                   + file_extension, format='wav')
            # wav_lower.export(save_path + "/" + cl + "/" + specimen + "/" + filename + "_lower"
            #                  + file_extension, format='wav')

            # pitch shift
            tones = [-3, 3]
            for n in range(len(tones)):

                # transpose and save
                transposed = librosa.effects.pitch_shift(wav2, sr, n_steps=float(tones[n]))
                soundfile.write(save_path + "/" + cl + "/" + specimen + "/" + filename +
                                "_transposed_" + str(n) + file_extension, transposed, sr)

            # time stretch
            rates = [0.9, 1.1]
            for n in range(len(rates)):

                stretched = librosa.effects.time_stretch(wav2, rate=rates[n])

                if len(stretched) < 48115:
                    temp = np.zeros(48115)
                    temp[:stretched.shape[0]] = stretched
                    stretched = temp
                elif len(stretched) > 48115:
                    stretched = stretched[:48115]

                soundfile.write(save_path + "/" + cl + "/" + specimen + "/" + filename +
                                "_stretched_" + str(n) + file_extension, stretched, sr)

            count += 1

