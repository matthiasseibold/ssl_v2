import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import librosa

plot_audio_labels = True
extract_frames = False

# setup paths
path_labels = "../labels/chiseling/"
root_frames = "F:/SSL/experiment_3d/output/chiseling/heatmaps/"

files = os.listdir(path_labels)
for f in files:

    filename, extension = os.path.splitext(f)
    print("Processing file: " + filename)

    # get events
    with open(path_labels + f, "r", encoding="utf-8") as ftxt:
        labels = ftxt.readlines()
        cleaned = [item.strip() for item in labels]
        time_steps = np.asarray(cleaned, dtype=float)

    print(str(len(cleaned)) + " labels found")

    if plot_audio_labels:
        # Load audio file
        audio_file = "F:/SSL/experiment_3d/output/chiseling/" + filename + ".wav"
        y, sr = librosa.load(audio_file, sr=None)  # y = waveform, sr = sample rate

        # Plot waveform
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title("Audio waveform with vertical lines")

        # Add vertical lines at each time step
        for t in time_steps:
            plt.axvline(x=t, color='r', linestyle='--', alpha=0.8)

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    if extract_frames:
        # Input video
        video_file = root_frames + filename + ".avi"

        # Output folder for extracted frames
        output_dir = root_frames + filename + "_extracted_frames"
        os.makedirs(output_dir, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_file}")

        # Get video FPS to calculate frame numbers
        # fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 25
        print(f"FPS: {fps}")

        # Loop through each time step
        for t in time_steps:
            frame_num = int(round(t * fps))  # Convert seconds to frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)  # Seek to frame

            ret, frame = cap.read()
            if not ret:
                print(f"Could not read frame at {t} seconds (frame {frame_num})")
                continue

            # Save frame as image
            frame_filename = os.path.join(output_dir, f"frame_{frame_num:06d}.png")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame at {t}s → {frame_filename}")

        # Release video
        cap.release()
