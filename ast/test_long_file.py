import os

import numpy as np
import matplotlib.pyplot as plt
import librosa
import evaluate
from datasets import Dataset, Audio, ClassLabel, Features
from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report, confusion_matrix

# init
root = "F:/datasets/ssl_v2/long_file_chiseling"
file = "1_016_Movie2D_heatmap"

test_y = np.load("data_ast/chiseling_long/test_y_" + file + ".npy")

wav_snippets = os.listdir(root + "/" + file)
test_x = [root + "/" + file + "/" + item for item in wav_snippets]

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
pretrained_model = "runs/best_model_chiseling"
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

# Configure training run with TrainingArguments class
training_args = TrainingArguments(
    output_dir="runs/ast_classifier",
    logging_dir="./logs/ast_classifier",
    # report_to="tensorboard",
    learning_rate=5e-5,  # Learning rate
    push_to_hub=False,
    num_train_epochs=10,  # Number of epochs
    per_device_train_batch_size=8,  # Batch size per device
    eval_strategy="epoch",  # Evaluation strategy
    save_strategy="epoch",
    eval_steps=1,
    save_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_strategy="steps",
    logging_steps=20,
)

accuracy = evaluate.load("accuracy")
recall = evaluate.load("recall")
precision = evaluate.load("precision")
f1 = evaluate.load("f1")

AVERAGE = "macro" if config.num_labels > 2 else "binary"

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    predictions = np.argmax(logits, axis=1)
    metrics = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    metrics.update(precision.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    metrics.update(recall.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    metrics.update(f1.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    return metrics

# Setup the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics,  # Use the metrics function from above
)

# trainer.evaluate()
predictions = trainer.predict(test_dataset=dataset_test)

# these are the predictions for every consecutive window of the long file
y_pred = predictions.predictions.argmax(axis=1)

for i in range(len(y_pred)):
    print("Predicted: " + str(y_pred[i]) + " --- Ground Truth: " + str(test_y[i]))

for i in range(len(y_pred)):
    if test_y[i] == 1 and test_y[i-1] == 0:
        print("Ground Truth: Peak detected at: " + str(0.15 + i * 0.02) + " s")

    if y_pred[i] == 1 and y_pred[i-1] == 0:
        print("Predictions: Peak detected at: " + str(0.15 + i * 0.02) + " s")