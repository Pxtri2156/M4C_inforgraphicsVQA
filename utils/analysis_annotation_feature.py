import os
import numpy as np


info_path = "/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/extracted_output/10030_info.npy"
path = "/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/extracted_output/10022.npy"
# output_screen_path = "/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/output_screen/annotation_textvqa.txt"

# output_screen_file = open(output_screen_path, 'w')

info_feature = np.load(info_path, allow_pickle=True) 
info_feature = info_feature[()]
print("key of info feature: ", info_feature.keys())
print("No bbox: ", info_feature["bbox"].shape)
print("No object: ", info_feature["objects"].shape)
print("No cls_prob: ", info_feature["cls_prob"].shape)

print(info_feature)
