import numpy as np
import evaluate
from datasets import Dataset, Audio, ClassLabel, Features
from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report, confusion_matrix

print("loading data")
test_x = np.load("data_ast/chiseling/test_x.npy").tolist()
test_y = np.load("data_ast/chiseling/test_y.npy").tolist()

print("Test data length: " + str(len(test_x)))
print("Test labels length: " + str(len(test_y)))
print("")

# Define class labels
class_labels = ClassLabel(names=["Healthy", "Idle", "Zenker"])

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
feature_extractor.mean = -1.1509622
feature_extractor.std = 3.5340312

dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
dataset_test = dataset_test.rename_column("audio", "input_values")

# w/o augmentations on the test set
dataset_test.set_transform(preprocess_audio, output_all_columns=False)

# Load configuration from the pretrained model
pretrained_model = "runs/ast_classifier/checkpoint-2562"
config = ASTConfig.from_pretrained(pretrained_model)

# Update configuration with the number of labels in our dataset
config.num_labels = 3
label2id = {
    "Healthy": 0,
    "Idle": 1,
    "Zenker": 2
}
config.label2id = label2id
config.id2label = {v: k for k, v in label2id.items()}

# Initialize the model with the updated configuration
model = ASTForAudioClassification.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)
model.init_weights()

# Configure training run with TrainingArguments class
training_args = TrainingArguments(
    output_dir="../runs/ast_classifier",
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
    print(metrics)
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

y_pred = predictions.predictions.argmax(axis=1)
y_true = predictions.label_ids

print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
print(cm)

print('Classification Report')
cr = classification_report(y_true, y_pred, target_names=['Healthy', 'Idle', 'Zenker'])
print(cr)
