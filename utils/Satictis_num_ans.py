import json 

train_path = "/mlcv/Databases/VN_InfographicVQA/train/VietInfographicVQA_train_v1.0.json"
val_path = "/mlcv/Databases/VN_InfographicVQA/val/VietInfographicVQA_val_v1.0.json"
train_fi  = open(train_path,)
val_fi  = open(val_path,)
train_data  = json.load(train_fi)
train_data = train_data['data']
val_data = json.load(val_fi)
val_data = val_data['data']
train_fi.close()
val_fi.close()

dic_train = {}
dic_val = {}

for sample in train_data:
    len_sample = len(sample['answers'])
    if len_sample not in dic_train.keys():
        dic_train[len_sample] = 1
    else:
        dic_train[len_sample] +=1    
print('train answer no: ', dic_train)

for sample in val_data:
    len_sample = len(sample['answers'])
    if len_sample not in dic_val.keys():
        dic_val[len_sample] = 1
    else:
        dic_val[len_sample] +=1  
print('val answer no: ', dic_val)