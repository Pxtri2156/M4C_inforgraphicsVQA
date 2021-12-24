import json
import argparse
import  sys
import os

sys.path.append("./")
from utils.get_dic_question_id import create_dic_question_id

DICT_QID = {}

def convert_gt(path, output):
    set_name = ['train', 'val', 'test']
    for i in range(len(set_name)):
        print("Processing, ", set_name[i])
        annot_path = path.replace("change", set_name[i])
        annot_fi = open(annot_path)
        data = json.load(annot_fi)
        for j in range(len(data['data'])):
            data['data'][j]['questionId'] = DICT_QID[data['data'][j]['questionId']]
        annot_fi.close()
        outpath = os.path.join(output, "VietInfographicVQA_change_v1.0_converted.json".replace('change', set_name[i] ))
        out_fi = open(outpath, 'w')
        json.dump(data, out_fi, indent=4, ensure_ascii=False )
        print("Saved new annotations at ", outpath)
        out_fi.close()


def main(args):
    convert_gt(args.path, args.new_folder)
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", 
        default="/mlcv/Databases/VN_InfographicVQA/change/VietInfographicVQA_change_v1.0.json", 
        type=str, 
    )
    parser.add_argument(
        "--new_folder", 
        default="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/submission/ViInfographicVQA/gt", 
        type=str, 
    )
    return parser.parse_args()

if __name__ == "__main__":
    path = "/mlcv/Databases/VN_InfographicVQA/change/VietInfographicVQA_change_v1.0.json"
    DICT_QID = create_dic_question_id(path)
    args = get_parser()
    main(args)