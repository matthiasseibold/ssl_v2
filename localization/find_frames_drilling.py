import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import librosa

plot_audio_labels = False
extract_frames = True

# setup paths
path_labels = "../labels/drilling/"
root_frames = "F:/SSL/experiment_3d/output/drilling/heatmaps/"

files = os.listdir(path_labels)
for f in files:

    filename, extension = os.path.splitext(f)
    print("Processing file: " + filename)

    # get events
    with open(path_labels + f, "r", encoding="utf-8") as ftxt:
        labels = ftxt.readlines()
        cleaned = [item.strip() for item in labels]
        cleaned = [item.split(",") for item in labels]
        time_steps = np.asarray(cleaned, dtype=float)

    print(str(len(cleaned)) + " labels found")

    if plot_audio_labels:
        # Load audio file
        audio_file = "F:/SSL/experiment_3d/output/drilling/" + filename + ".wav"
        y, sr = librosa.load(audio_file, sr=None)  # y = waveform, sr = sample rate

        # Plot waveform
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title("Audio waveform with vertical lines")

        # Add vertical lines at each time step
        for t in time_steps.reshape(-1):
            plt.axvline(x=t, color='r', linestyle='--', alpha=0.8)

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    if extract_frames:
        # Input video
        video_path = root_frames + filename + ".avi"

        # Output folder for extracted frames
        output_dir = root_frames + filename + "_extracted_frames"

        # --- Prepare output folder ---
        os.makedirs(output_dir, exist_ok=True)

        # --- Load video ---
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video FPS: {fps}, Total frames: {total_frames}")

        for j in range(time_steps.shape[0]):

            start_stop = time_steps[j]

            start_time = start_stop[0]  # seconds
            end_time = start_stop[1]  # seconds
            frame_step = 5  # extract every 5th frame

            # --- Compute frame range ---
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            # --- Set starting position ---
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_idx = start_frame
            saved_count = 0

            while cap.isOpened() and frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save every 10th frame
                if (frame_idx - start_frame) % frame_step == 0:
                    filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
                    cv2.imwrite(filename, frame)
                    saved_count += 1

                frame_idx += 1

        cap.release()
        print(f"✅ Done! Saved {saved_count} frames to '{output_dir}'")
