import os
import random
import numpy as np
from LoadSpectrograms import load_data_single_folder

dataset_root = "F:/spectrograms/chiseling"
test_root = "F:/spectrograms/chiseling"
normalize = True

train_folders = ["/peak/1_011_Movie2D_heatmap/",
                 "/peak/1_012_Movie2D_heatmap/",
                 "/peak/1_013_Movie2D_heatmap/",
                 "/peak/1_014_Movie2D_heatmap/",
                 "/peak/1_015_Movie2D_heatmap/",
                 "/nopeak/1_011_Movie2D_heatmap/",
                 "/nopeak/1_012_Movie2D_heatmap/",
                 "/nopeak/1_013_Movie2D_heatmap/",
                 "/nopeak/1_014_Movie2D_heatmap/",
                 "/nopeak/1_015_Movie2D_heatmap/"]
test_folders = ["/peak/1_016_Movie2D_heatmap/",
                "/nopeak/1_016_Movie2D_heatmap/"]

train_folders = [dataset_root + item for item in train_folders]
test_folders = [dataset_root + item for item in test_folders]

print(len(train_folders))
print(len(test_folders))

print('TRAIN FOLDERS')
print(train_folders)
print('')
print('TEST FOLDERS')
print(test_folders)
print('')

for idx, f in enumerate(train_folders):
    x, y = load_data_single_folder([f], normalize)
    if idx == 0:
        train_x = x
        train_y = y
    else:
        train_x = np.append(train_x, x, axis=0)
        train_y = np.append(train_y, y, axis=0)

for idx, f in enumerate(test_folders):
    x, y = load_data_single_folder([f], normalize)
    if idx == 0:
        test_x = x
        test_y = y
    else:
        test_x = np.append(test_x, x, axis=0)
        test_y = np.append(test_y, y, axis=0)

print("Training data shape: " + str(train_x.shape))
print("Training labels shape: " + str(train_y.shape))
print("Test data shape: " + str(test_x.shape))
print("Test labels shape: " + str(test_y.shape))
print("")

np.save("../data/train_x.npy", train_x)
np.save("../data/test_x.npy", test_x)
np.save("../data/train_y.npy", np.eye(2)[train_y])
np.save("../data/test_y.npy", np.eye(2)[test_y])
