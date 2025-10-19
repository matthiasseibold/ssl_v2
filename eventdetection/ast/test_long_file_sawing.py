import os
import numpy as np
import evaluate
from datasets import Dataset, Audio, ClassLabel, Features
from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification, TrainingArguments, Trainer

verbose = True
relaxed_condition = False

# init
root = "F:/datasets/ssl_v2/long_file_sawing"
files = ["1_006_Movie2D_heatmap",
         "1_007_Movie2D_heatmap",
         "1_008_Movie2D_heatmap",
         "1_009_Movie2D_heatmap",
         "1_010_Movie2D_heatmap",]
fold = "fold1"

for count, file in enumerate(files):

    test_y = np.load("data_ast/sawing_long_" + file + "/test_y.npy")

    wav_snippets = os.listdir(root + "_" + file + "/" + file)
    test_x = [root + "_" + file + "/" + file + "/" + item for item in wav_snippets]

    # Define class labels
    class_labels = ClassLabel(names=["nosawing", "sawing"])

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
    pretrained_model = "runs/best_model_sawing_" + fold
    config = ASTConfig.from_pretrained(pretrained_model)

    # Update configuration with the number of labels in our dataset
    config.num_labels = 2
    label2id = {
        "nosawing": 0,
        "sawing": 1
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

    FN = 0
    FP = 0
    TP = 0
    gt_onsets = 0
    tp_log = np.zeros(len(y_pred))
    fp_log = np.zeros(len(y_pred))

    for i in range(len(y_pred) - 1):

        cond_gt = test_y[i] == 1 and test_y[i - 1] == 0
        cond_pred = y_pred[i - 2] == 0 and y_pred[i - 1] == 0 and y_pred[i] == 1 and y_pred[i + 1] == 1
        # we count the detection as true positive, if there is one frame offset (if it's detected one frame too early or too late)
        cond_pred_relaxed = (y_pred[i] == 1 and y_pred[i - 1] == 0) or (y_pred[i - 1] == 1 and y_pred[i - 2] == 0) or (
                    y_pred[i + 1] == 1 and y_pred[i] == 0)

        if verbose:
            print("Frame number: " + str(i) + " --- Predicted: " + str(y_pred[i]) + " --- Ground Truth: " + str(
                test_y[i]))
            if cond_gt:
                print("Ground Truth: Peak detected at: " + str(0.15 + i * 0.02) + " s")
            if cond_pred:
                print("Predictions: Peak detected at: " + str(0.15 + i * 0.02) + " s")

        if relaxed_condition:
            if cond_gt:
                gt_onsets += 1
            if cond_pred and not cond_gt:
                fp_log[i] = 1
                # we don't count it if we had a true positive before (relaxed condition)
                if tp_log[i - 1] == 1:
                    fp_log[i] = 0
            if cond_pred_relaxed and cond_gt:
                TP += 1
                tp_log[i] = 1
                # also, we don't count the false positive if in the frame directly after a true positive is detected (relaxed condition)
                if fp_log[i - 1] == 1:
                    fp_log[i - 1] = 0
        else:
            if cond_gt:
                gt_onsets += 1
            if cond_pred and not cond_gt:
                fp_log[i] = 1
            if cond_pred and cond_gt:
                tp_log[i] = 1

    FN = gt_onsets - np.sum(tp_log)
    FP = np.sum(fp_log)
    TP = np.sum(tp_log)

    if count == 0:
        FN_all = FN
        FP_all = FP
        TP_all = TP
    else:
        FN_all = FN_all + FN
        FP_all = FP_all + FP
        TP_all = TP_all + TP

FN = FN_all
FP = FP_all
TP = TP_all

print("")
print("Final results: ")
print("FN: " + str(FN))
print("FP: " + str(FP))
print("TP: " + str(TP))
print("")
print("Precision: " + str(TP / (TP + FP)))
print("Recall: " + str(TP / (TP + FN)))
print("F1: " + str(2 * TP / (2 * TP + FN + FP)))