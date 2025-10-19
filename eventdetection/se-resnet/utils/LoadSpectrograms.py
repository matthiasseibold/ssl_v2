import os
import numpy as np

# dataset statistics
mean = -25.694532
std = 8.431269

def load_data_single_folder(train_folders, normalize=False):

    train_data = np.zeros(0)
    train_y = 0

    for i, train_f in enumerate(train_folders):

        print("Loading folder: " + train_f)

        files = os.listdir(train_f)

        for j, file in enumerate(files):
            temp = np.load(train_f + "/" + file)
            temp = np.reshape(temp, (1, temp.shape[0], temp.shape[1], 1))
            temp = temp[:, :, :]

            if normalize:
                temp = (temp - mean) / std

            if i == 0 and j == 0:
                train_data = temp
                if "nopeak" in train_f:
                    train_y = 0
                else:
                    train_y = 1
            else:
                train_data = np.append(train_data, temp, axis=0)
                if "nopeak" in train_f:
                    train_y = np.append(train_y, 0)
                else:
                    train_y = np.append(train_y, 1)

    return train_data, train_y
