import os
import numpy as np

path = "/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/data/datasets/textvqa/defaults/annotations/imdb_train_ocr_en.npy"
# output_screen_path = "/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/output_screen/annotation_textvqa.txt"

# output_screen_file = open(output_screen_path, 'w')

imdb_test_ocr_en = np.load(path, allow_pickle=True) 
print('len: ', len(imdb_test_ocr_en))
print('element 0: ', imdb_test_ocr_en[0])
# print('element 1 have key: ', imdb_test_ocr_en[1].keys())


example_annotation = imdb_test_ocr_en[30]
for key in example_annotation.keys():
    print("{}: {}".format(key, example_annotation[key]))

# for element in imdb_test_ocr_en:
#     print(element.keys())
# print('element 1: ', imdb_test_ocr_en[1])


# print("imdb test ocr en: ", imdb_test_ocr_en)
