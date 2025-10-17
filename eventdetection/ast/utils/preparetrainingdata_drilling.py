import os
import numpy as np

dataset_root = "F:/datasets/ssl_v2"
save_path = "../data_ast/drilling_fold1/"

train_recordings = ["1_017_Movie2D_heatmap/",
                    "1_018_Movie2D_heatmap/",
                    "1_019_Movie2D_heatmap/",
                    "1_020_Movie2D_heatmap/"]

test_recordings = ["1_021_Movie2D_heatmap/",
                    "1_022_Movie2D_heatmap/"]

train_folders = ["/drilling_ast/" + item for item in train_recordings] + ["/nodrilling_ast/" + item for item in train_recordings]
test_folders = ["/drilling_ast/" + item for item in test_recordings] + ["/nodrilling_ast/" + item for item in test_recordings]

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

    if "nodrilling" in f:
        y = 0 * np.ones(len(files))
    elif "drilling" in f:
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

    if "nodrilling" in f:
        y = 0 * np.ones(len(files))
    elif "drilling" in f:
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
