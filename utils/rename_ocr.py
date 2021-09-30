import os
path = "./"

info_list = []
info_info_list = []

for file_name in os.listdir(path):
    print('Processing file', file_name)
    if 'npy' in file_name:
        lst = file_name.split("_")
        if 'info' in lst:
            info_info_list.append(file_name)
        else:
            info_list.append(file_name)
    else:
        continue

# print(info_info_list)
# print(info_info_list)

for file_name in info_list:
    new_name = file_name.replace("_info.npy", ".npy")
    print("new name: ", new_name)
    os.rename(file_name, new_name)

for file_name in info_info_list:
    new_name = file_name.replace("_info.npy", ".npy")
    print("new name: ", new_name)
    os.rename(file_name, new_name)
