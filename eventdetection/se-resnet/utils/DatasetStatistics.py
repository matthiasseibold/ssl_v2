import os
import numpy as np


def compute_statistics(folders):

    mean_array = 0
    std_array = 0

    for i, train_f in enumerate(folders):

        print("Loading folder: " + train_f)

        files = os.listdir(train_f)

        for j, file in enumerate(files):
            temp = np.load(train_f + "/" + file, allow_pickle=True)
            temp = np.reshape(temp, (1, temp.shape[0], temp.shape[1], 1))
            temp = temp[:, :, :210]
            if i == 0 and j == 0:
                mean_array = np.mean(temp)
                std_array = np.std(temp)
            else:
                mean_array = np.append(mean_array, np.mean(temp))
                std_array = np.append(std_array, np.std(temp))

    return np.mean(mean_array), np.mean(std_array)


if __name__ == '__main__':

    root_dir = "F:\spectrograms\chiseling"
    train_folders = []
    classes = os.listdir(root_dir)
    for cl in classes:
        specimens = os.listdir(root_dir + "/" + cl)
        for specimen in specimens:
            train_folders.append(root_dir + "/" + cl + "/" + specimen)

    mean, std = compute_statistics(train_folders)

    print(mean)
    print(std)
