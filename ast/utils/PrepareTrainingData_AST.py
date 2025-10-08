import os
import random
import numpy as np

dataset_root = "F:/datasets/New_SwallowSet/Test"
test_root =  "F:/datasets/New_SwallowSet/Test"

train_folders = []
test_folders = []
classes = os.listdir(dataset_root)
for cl in classes:
    specimens = os.listdir(dataset_root + "/" + cl)
    train_specimens = random.sample(specimens, int(len(specimens) * 0.8))
    test_specimens = list(set(specimens) - set(train_specimens))
    for specimen in train_specimens:
        train_folders.append(dataset_root + "/" + cl + "/" + specimen + "/")
    for specimen in test_specimens:
        test_folders.append(test_root + "/" + cl + "/" + specimen + "/")

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

    if "Healthy" in f:
        y = 0 * np.ones(len(files))
    elif "Idle" in f:
        y = 1 * np.ones(len(files))
    else:
        y = 2 * np.ones(len(files))


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

    if "Healthy" in f:
        y = 0 * np.ones(len(files))
    elif "Idle" in f:
        y = 1 * np.ones(len(files))
    else:
        y = 2 * np.ones(len(files))

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

if not os.path.exists("../data_ast/"):
    os.makedirs("../data_ast/")

np.save("../data_ast/train_x.npy", train_x)
np.save("../data_ast/test_x.npy", test_x)
np.save("../data_ast/train_y.npy", train_y)
np.save("../data_ast/test_y.npy", test_y)
