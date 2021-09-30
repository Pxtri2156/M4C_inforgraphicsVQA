
import os
import json
import cv2

# ocr_path = "/mlcv/Databases/DocVQA_2020-21/task_3/train/ocr_results/20467.json"
# ocr_file = open(ocr_path, )
# ocr_data = json.load(ocr_file)
# print("keys: ", ocr_data.keys())

# img_path ='/mlcv/Databases/DocVQA_2020-21/task_3/train/infographicVQA_train_v1.0_images/20467.jpeg'

# img = cv2.imread(img_path)
# cv2.imwrite('20467.jpeg', img)
 
path = "/mlcv/Databases/DocVQA_2020-21/task_3/train/ocr_results/"

for file_name in os.listdir(path):
    file_path = os.path.join(path, file_name)
    ocr_file = open(file_path, )
    ocr_data = json.load(ocr_file)
    if 'WORD' not in ocr_data.keys():
        print(file_name)
