import argparse
import json
from os import openpty

def create_dic_question_id(path):
    set_name = ['train', 'val', 'test']
    dic_qid = {}
    for i in range(len(set_name)):
        print("Processing, ", set_name[i])
        annot_path = path.replace("change", set_name[i])
        annot_fi = open(annot_path)
        data = json.load(annot_fi)
        data = data['data']
        for sample in data:
            questionId = sample['questionId']
            img_id = sample['image'].split('/')[-1].split('.')[0]
            new_id = int(str(i+1) + img_id + '00')
            while new_id in dic_qid.keys():
                new_id += 1
            dic_qid[questionId] = new_id
            dic_qid[new_id] = questionId
        annot_fi.close()
    return dic_qid

def main(args):
    dic_qid = create_dic_question_id(args.path) 
    print(dic_qid)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", 
        default="/mlcv/Databases/VN_InfographicVQA/change/VietInfographicVQA_change_v1.0.json", 
        type=str, 
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    main(args)