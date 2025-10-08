import os
import random
import numpy as np

test_root = "../long_file_windows/"

test_folders = []
classes = os.listdir(test_root)
for cl in classes:
    specimens = os.listdir(test_root + "/" + cl)
    for specimen in specimens:
        test_folders.append(test_root + "/" + cl + "/" + specimen + "/")

print(len(test_folders))

print('TEST FOLDERS')
print(test_folders)
print('')

test_x = []

for idx, f in enumerate(test_folders):

    files = os.listdir(f)

    if f.startswith("../"):
        f = f[1:]

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

print("Test data length: " + str(len(test_x)))
print("Test labels length: " + str(len(test_y)))
print("")

if not os.path.exists("../data_ast_long/"):
    os.makedirs("../data_ast_long/")

np.save("../data_ast_long/test_x.npy", test_x)
np.save("../data_ast_long/test_y.npy", test_y)
