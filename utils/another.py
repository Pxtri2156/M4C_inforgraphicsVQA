import numpy as np

path = "env_variable/data/datasets/inforgraphicvqa/defaults/annotations/vqa_train_en.npy"

data = np.load(path, allow_pickle=True)
print("Load successful")