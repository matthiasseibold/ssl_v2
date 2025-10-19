import numpy as np
import torch
from datasets import Dataset, Audio, ClassLabel, Features
from transformers import ASTFeatureExtractor


print("loading data")
train_x = np.load("../data_ast/chiseling/train_x.npy").tolist()
train_y = np.load("../data_ast/chiseling/train_y.npy").tolist()

print("Training data length: " + str(len(train_x)))
print("Training labels length: " + str(len(train_y)))
print("")

# Define class labels
class_labels = ClassLabel(names=["nopeak", "peak"])

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


# we define which pretrained model we want to use and instantiate a feature extractor
pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)

# we save model input name and sampling rate for later use
model_input_name = feature_extractor.model_input_names[0]  # key -> 'input_values'

def preprocess_audio(batch):
    wavs = [audio['array'] for audio in batch['input_values']]
    inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}
    return output_batch

# Apply the transformation to the dataset
dataset_train = dataset_train.rename_column("audio", "input_values")  # rename audio column
dataset_train.set_transform(preprocess_audio, output_all_columns=False)

# calculate values for normalization
feature_extractor.do_normalize = False  # we set normalization to False in order to calculate the mean + std of the dataset
mean = []
std = []

# we use the transformation w/o augmentation on the training dataset to calculate the mean + std
dataset_train.set_transform(preprocess_audio, output_all_columns=False)
for i, (audio_input, labels) in enumerate(dataset_train):
    print("Processing sample: " + str(i) + "/" + str(len(train_x)))
    cur_mean = torch.mean(dataset_train[i][audio_input])
    cur_std = torch.std(dataset_train[i][audio_input])
    mean.append(cur_mean)
    std.append(cur_std)
feature_extractor.do_normalize = True
print(np.mean(mean))
print(np.mean(std))
#####
