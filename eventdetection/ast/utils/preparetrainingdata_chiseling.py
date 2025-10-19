import os
import numpy as np

dataset_root = "F:/datasets/ssl_v2"
save_path = "../data_ast/chiseling_fold3/"

train_folders = ["/peak_ast/1_013_Movie2D_heatmap/",
                 "/peak_ast/1_014_Movie2D_heatmap/",
                 "/peak_ast/1_015_Movie2D_heatmap/",
                 "/peak_ast/1_016_Movie2D_heatmap/",
                 "/nopeak_ast/1_013_Movie2D_heatmap/",
                 "/nopeak_ast/1_014_Movie2D_heatmap/",
                 "/nopeak_ast/1_015_Movie2D_heatmap/",
                 "/nopeak_ast/1_016_Movie2D_heatmap/"]
test_folders = ["/peak_ast/1_011_Movie2D_heatmap/",
                 "/peak_ast/1_012_Movie2D_heatmap/",
                "/nopeak_ast/1_011_Movie2D_heatmap/",
                "/nopeak_ast/1_012_Movie2D_heatmap/"]

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

train_x = []
test_x = []

for idx, f in enumerate(train_folders):

    files = os.listdir(f)

    if "nopeak" in f:
        y = 0 * np.ones(len(files))
    elif "peak" in f:
        y = 1 * np.ones(len(files))


    for i, current_file in enumerate(files):
        if i == 0:
            temp_x = [f + current_file]
        else:
            temp_x.append(f + current_file)

    if idx == 0:
        train_y = y
        train_x = temp_x
    else:
        train_y = np.append(train_y, y)
        train_x.extend(temp_x)


for idx, f in enumerate(test_folders):

    files = os.listdir(f)

    if "nopeak" in f:
        y = 0 * np.ones(len(files))
    elif "peak" in f:
        y = 1 * np.ones(len(files))

    for i, current_file in enumerate(files):
        if i == 0:
            temp_x = [f + current_file]
        else:
            temp_x.append(f + current_file)

    if idx == 0:
        test_y = y
        test_x = temp_x
    else:
        test_y = np.append(test_y, y)
        test_x.extend(temp_x)

print("Training data length: " + str(len(train_x)))
print("Training labels length: " + str(len(train_y)))
print("Test data length: " + str(len(test_x)))
print("Test labels length: " + str(len(test_y)))
print("")

os.makedirs(save_path, exist_ok=True)

np.save(save_path + "train_x.npy", train_x)
np.save(save_path + "test_x.npy", test_x)
np.save(save_path + "train_y.npy", train_y)
np.save(save_path + "test_y.npy", test_y)
