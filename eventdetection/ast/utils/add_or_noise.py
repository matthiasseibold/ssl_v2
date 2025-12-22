import os
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
import math

root = "F:/datasets/ssl_v2"
folders = os.listdir(root)
filtered = [f for f in folders if "long_file" in f]

save_path = root + "/test_or_noise_50_percent"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for f in filtered:

    save_path_full = save_path + "/" + f
    if not os.path.exists(save_path_full):
        os.makedirs(save_path_full)

    subfolder = os.listdir(os.path.join(root, f))
    files = os.listdir(root + "/" + f + "/" + subfolder[0])

    print("Processing file: " + str(subfolder[0]))

    for file in files:

        if file.endswith(".wav"):

            # ---- Paths ----
            audio1_path = root + "/" + f + "/" + subfolder[0] + "/" + file
            audio2_path = r"F:/datasets/ssl_v2/or_background_noise/632733__nicotep__operation-chirurgicale-oaxaca-16fev20176pm.wav"

            # ---- Load first audio ----
            audio1, sr1 = sf.read(audio1_path)
            if audio1.ndim > 1:
                audio1 = np.mean(audio1, axis=1)  # mono

            length1 = len(audio1)

            # ---- Load second audio ----
            audio2, sr2 = sf.read(audio2_path)
            if audio2.ndim > 1:
                audio2 = np.mean(audio2, axis=1)

            # ---- Resample second audio if needed ----
            if sr2 != sr1:
                gcd = math.gcd(sr2, sr1)
                up = sr1 // gcd
                down = sr2 // gcd
                audio2 = resample_poly(audio2, up, down)

            # ---- Take random segment from second audio ----
            if len(audio2) < length1:
                raise ValueError("Background audio is shorter than foreground audio")

            start = np.random.randint(0, len(audio2) - length1 + 1)
            audio2_segment = audio2[start:start + length1]

            # ---- Mix with amplitude scaling ----
            noise_scale = 0.5
            mixed = audio1 + noise_scale * audio2_segment

            # ---- Prevent clipping ----
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0:
                mixed = mixed / max_val

            # ---- Save result ----
            sf.write(save_path_full + "/" + file, mixed, sr1)

            print("Mixing complete. Saved file: " + str(file))


