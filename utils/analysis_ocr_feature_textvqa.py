import numpy as np
import os

path = "/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/lmdb_ocr/textVQA"

for name in os.listdir(path):
    file_path = os.path.join(path, name)
    ocr_info_feature = np.load(file_path, allow_pickle=True)
    print(ocr_info_feature)