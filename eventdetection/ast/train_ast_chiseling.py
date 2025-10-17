import numpy as np
import evaluate
from datasets import Dataset, Audio, ClassLabel, Features
from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification, TrainingArguments, Trainer
from audiomentations import Compose, AddGaussianSNR, GainTransition, Gain, ClippingDistortion, TimeStretch, PitchShift
from sklearn.metrics import classification_report, confusion_matrix

print("loading data")
train_x = np.load("data_ast/chiseling_fold2/train_x.npy").tolist()
train_y = np.load("data_ast/chiseling_fold2/train_y.npy").tolist()
test_x = np.load("data_ast/chiseling_fold2/test_x.npy").tolist()
test_y = np.load("data_ast/chiseling_fold2/test_y.npy").tolist()

print("Training data length: " + str(len(train_x)))
print("Training labels length: " + str(len(train_y)))
print("Test data length: " + str(len(test_x)))
print("Test labels length: " + str(len(test_y)))
print("")

classes = ["nopeak", "peak"]

# Define class labels
class_labels = ClassLabel(names=classes)

SAMPLING_RATE = 16000

# Define features with audio and label columns
features = Features({
    "audio": Audio(),  # Define the audio feature
    "labels": class_labels  # Assign the class labels
})

# Construct the dataset from a dictionary
dataset_train = Dataset.from_dict({
    "audio": train_x,
    "labels": train_y,  # Corresponding labels for the audio files
}, features=features)
dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

dataset_test = Dataset.from_dict({
    "audio": test_x,
    "labels": test_y,  # Corresponding labels for the audio files
}, features=features)
dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

# we define which pretrained model we want to use and instantiate a feature extractor
pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)

# we save model input name and sampling rate for later use
model_input_name = feature_extractor.model_input_names[0]  # key -> 'input_values'

# computed with datasetstats.py
feature_extractor.mean = -0.18241186
feature_extractor.std = 1.117633

# Augmentations
audio_augmentations = Compose([
    AddGaussianSNR(min_snr_db=10, max_snr_db=20),
    Gain(min_gain_db=-6, max_gain_db=6),
    GainTransition(min_gain_db=-6, max_gain_db=6, min_duration=0.01, max_duration=0.3, duration_unit="fraction"),
    ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=30, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2),
    PitchShift(min_semitones=-4, max_semitones=4),
], p=0.8, shuffle=True)

def preprocess_audio(batch):
    wavs = [audio['array'] for audio in batch['input_values']]
    # inputs are spectrograms as torch.tensors now
    inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")

    output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}
    return output_batch

def preprocess_audio_with_transforms(batch):
    # we apply augmentations on each waveform
    wavs = [audio_augmentations(audio["array"], sample_rate=SAMPLING_RATE) for audio in batch["input_values"]]
    inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")

    output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

    return output_batch

# Cast the audio column to the appropriate feature type and rename it
dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
dataset_train = dataset_train.rename_column("audio", "input_values")  # rename audio column

dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
dataset_test = dataset_test.rename_column("audio", "input_values")

# with augmentations on the training set
# dataset_train.set_transform(preprocess_audio_with_transforms, output_all_columns=False)
dataset_train.set_transform(preprocess_audio, output_all_columns=False)
# w/o augmentations on the test set
dataset_test.set_transform(preprocess_audio, output_all_columns=False)


# Load configuration from the pretrained model
config = ASTConfig.from_pretrained(pretrained_model)

# Update configuration with the number of labels in our dataset
config.num_labels = 2
label2id = {
    "nopeak": 0,
    "peak": 1,
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
    learning_rate=1e-5,  # Learning rate
    push_to_hub=False,
    num_train_epochs=5,  # Number of epochs
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
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics,  # Use the metrics function from above
)

trainer.train()

# get some statistics
# trainer.evaluate()
predictions = trainer.predict(test_dataset=dataset_test)

y_pred = predictions.predictions.argmax(axis=1)
y_true = predictions.label_ids

print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
print(cm)

print('Classification Report')
cr = classification_report(y_true, y_pred, target_names=classes)
print(cr)

# save best model
trainer.save_model('runs/best_model_chiseling')