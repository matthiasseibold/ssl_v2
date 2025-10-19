import numpy as np
import matplotlib.pyplot as plt
import librosa

import os
import numpy as np
import evaluate
from datasets import Dataset, Audio, ClassLabel, Features
from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification, TrainingArguments, Trainer

verbose = False
relaxed_condition = True

# init
root = "F:/datasets/ssl_v2/long_file_drilling"
files = ["1_018_Movie2D_heatmap"]
fold = "fold3"
relaxed_window = 10

# change font for matplotlib
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

def transition_check(y_pred, i, window=10):
    """
    Returns True if a 0→1 transition occurs at index i or
    within ±`window` frames around it.

    Parameters
    ----------
    y_pred : list or np.ndarray
        Sequence of predicted values (0s and 1s).
    i : int
        Current frame index to check.
    window : int
        Number of frames to look backward and forward (default: 10).

    Returns
    -------
    bool
        True if a 0→1 transition occurs at or near index i.
    """
    y_pred = np.asarray(y_pred)

    # Handle edge cases
    if len(y_pred) < 2 or i <= 0 or i >= len(y_pred):
        return False

    # Define window safely
    start = max(1, i - window)
    end = min(len(y_pred) - 1, i + window)

    # Check for any 0→1 transition within ±window frames
    for j in range(start, end + 1):
        if y_pred[j-1] == 0 and y_pred[j] == 1:
            return True

    return False

def has_one_nearby(arr, i, window=10):
    """
    Returns True if there is a 1 within ±`window` frames of index i.

    Parameters
    ----------
    arr : list or np.ndarray
        Sequence of 0s and 1s.
    i : int
        Current frame index to check.
    window : int
        Number of frames to look backward and forward (default: 10).

    Returns
    -------
    bool
        True if there is at least one 1 in the surrounding window.
    """
    arr = np.asarray(arr)

    # Check index validity
    if i < 0 or i >= len(arr):
        return False

    # Define safe window bounds
    start = max(0, i - window)
    end = min(len(arr), i + window + 1)

    # Check for any 1 in the window
    return np.any(arr[start:end] == 1)

def get_event_times(predictions, win_len=0.15, hop=0.02, threshold=0.5, use_center=False, rising_only=True):
    """
    Returns event times (in seconds) from sliding-window predictions.

    Parameters
    ----------
    predictions : array-like
        Sequence of predictions (binary or continuous).
    win_len : float
        Length of each analysis window in seconds.
    hop : float
        Hop length between consecutive windows in seconds.
    threshold : float
        Threshold for event detection if predictions are continuous.
    use_center : bool
        If True, report times at window centers; if False, use window starts.
    rising_only : bool
        If True, return only the start of each new event (rising edges).

    Returns
    -------
    event_times : np.ndarray
        Array of times (in seconds) where events occur.
    """

    predictions = np.asarray(predictions)
    binary_events = predictions >= threshold

    if rising_only:
        # detect rising edges (start of new event)
        rising_edges = np.where(np.diff(binary_events.astype(int)) == 1)[0] + 1
        indices = rising_edges
    else:
        # include all frames where event is active
        indices = np.where(binary_events)[0]

    if use_center:
        times = indices * hop + win_len / 2
    else:
        times = indices * hop  # window start

    return times

for count, file in enumerate(files):

    test_y = np.load("../data_ast/drilling_long_" + file + "/test_y.npy")

    wav_snippets = os.listdir(root + "_" + file + "/" + file)
    test_x = [root + "_" + file + "/" + file + "/" + item for item in wav_snippets]

    # Define class labels
    class_labels = ClassLabel(names=["nopeak", "peak"])

    SAMPLING_RATE = 16000

    # Define features with audio and label columns
    features = Features({
        "audio": Audio(),  # Define the audio feature
        "labels": class_labels  # Assign the class labels
    })

    # construct dataset
    dataset_test = Dataset.from_dict({
        "audio": test_x,
        "labels": test_y,  # Corresponding labels for the audio files
    }, features=features)
    dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    # we define which pretrained model we want to use and instantiate a feature extractor
    pretrained_model_fe = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model_fe)

    # we save model input name and sampling rate for later use
    model_input_name = feature_extractor.model_input_names[0]  # key -> 'input_values'

    def preprocess_audio(batch):
        wavs = [audio['array'] for audio in batch['input_values']]
        # inputs are spectrograms as torch.tensors now
        inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")

        output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}
        return output_batch

    # computed mean and std from training dataset
    feature_extractor.mean = -0.18241186
    feature_extractor.std = 1.117633

    dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
    dataset_test = dataset_test.rename_column("audio", "input_values")

    # w/o augmentations on the test set
    dataset_test.set_transform(preprocess_audio, output_all_columns=False)

    # Load configuration from the pretrained model
    pretrained_model = "../runs/best_model_drilling_" + fold
    config = ASTConfig.from_pretrained(pretrained_model)

    # Update configuration with the number of labels in our dataset
    config.num_labels = 2
    label2id = {
        "nopeak": 0,
        "peak": 1
    }
    config.label2id = label2id
    config.id2label = {v: k for k, v in label2id.items()}

    # Initialize the model with the updated configuration
    model = ASTForAudioClassification.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)
    model.init_weights()

    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")

    AVERAGE = "macro" if config.num_labels > 2 else "binary"

    # Setup the trainer
    trainer = Trainer(
        model=model,
        eval_dataset=dataset_test
    )

    # trainer.evaluate()
    predictions = trainer.predict(test_dataset=dataset_test)

    # these are the predictions for every consecutive window of the long file
    y_pred = predictions.predictions.argmax(axis=1)
    # np.save("drilling_ypred.npy", y_pred)

    FN = 0
    FP = 0
    TP = 0
    gt_onsets = np.zeros(len(y_pred))
    tp_log = np.zeros(len(y_pred))
    fp_log = np.zeros(len(y_pred))

    for i in range(len(y_pred)):

        cond_gt = test_y[i] == 1 and test_y[i - 1] == 0
        cond_pred = y_pred[i - 2] == 0 and y_pred[i - 1] == 0 and y_pred[i] == 1
        # we count the detection as true positive, if there is one frame offset (if it's detected one frame too early or too late)
        cond_pred_relaxed = transition_check(y_pred, i=i, window=relaxed_window)

        if verbose:
            print("Frame number: " + str(i) + " --- Predicted: " + str(y_pred[i]) + " --- Ground Truth: " + str(
                test_y[i]))
            if cond_gt:
                print("Ground Truth: Peak detected at: " + str(0.15 + i * 0.02) + " s")
            if cond_pred:
                print("Predictions: Peak detected at: " + str(0.15 + i * 0.02) + " s")

        if relaxed_condition:
            if cond_gt:
                gt_onsets[i] = 1
            if cond_pred and not cond_gt:
                fp_log[i] = 1
            if cond_pred_relaxed and cond_gt:
                TP += 1
                tp_log[i] = 1


    audio_file = "F:/SSL/experiment_3d/output/drilling/" + file + ".wav"
    waveform, sr = librosa.load(audio_file, sr=None)

    # get events
    path_labels = "../../../labels/drilling/"

    with open(path_labels + file + ".csv", "r", encoding="utf-8") as ftxt:
        labels = ftxt.readlines()
        cleaned = [item.strip() for item in labels]
        cleaned = [item.split(",") for item in labels]
        time_steps = np.asarray(cleaned, dtype=float)

    # Example ground truth and predicted event times (in the same time units as t)
    ground_truth_events = time_steps[:,0]
    predicted_events = get_event_times(tp_log) + 0.13  # adjust the timestamps for the window length - one sliding window
                                                       # (first onset will be detected in the end of the window)
    print("GT Onsets:")
    print(gt_onsets)
    print("")
    print("Original labels:")
    print(time_steps)
    print("")
    print("Predicted labels:")
    print(predicted_events)

    # Plot the waveform
    librosa.display.waveshow(waveform, sr=sr, alpha=0.6)

    # Parameters for arrows
    arrow_length = 0.3
    arrow_width = 0.02
    y_top = max(waveform) + 0.5  # position arrows just above waveform top
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
    # plt.title('Waveform with Ground Truth (red) and Predicted (green) Events')
    # plt.legend(['Waveform', 'Ground Truth Events', 'Predicted Events'])
    plt.show()