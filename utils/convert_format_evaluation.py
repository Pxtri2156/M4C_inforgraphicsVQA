import json
import argparse
import  sys
sys.path.append("./")
from utils.get_dic_question_id import create_dic_question_id
DICT_QID = {}

def convert_format_submission(input_path, output_path):
    fi = open(input_path, )
    out_fi = open(output_path, 'w', encoding="utf8")
    data = json.load(fi)
    new_data = []
    print(len(data))
    for element in data:
        question_id =  element['question_id']
        print("Processing question ", question_id )
        # print(type(element))
        new_data.append({
            "questionId": question_id,
            "answer": element['answer']
        })
    print("Done")
    json.dump(new_data, out_fi, indent=4, ensure_ascii=False)
    fi.close()
    out_fi.close()

def convert_format_submissionVN(input_path, output_path):
    # global DICT_QID
    fi = open(input_path, )
    out_fi = open(output_path, 'w', encoding="utf8")
    data = json.load(fi)
    new_data = []
    print(len(data))
    for element in data:
        question_id =  element['question_id']
        print("Processing encode question {} real question id{} ".format( question_id,  DICT_QID[question_id]) )
        ()
        # print(type(element))
        new_data.append({
            "questionId": DICT_QID[question_id],
            "answer": element['answer']
        })
    print("Done")
    json.dump(new_data, out_fi, indent=4, ensure_ascii=False)
    fi.close()
    out_fi.close()
    
def main(args):
    convert_format_submissionVN(args.input_path, args.output_path)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", 
        default="env_variable/report/inforgraphicVQA/inforgraphicvqa_run_test_2021-09-29T16:02:09.json", 
        type=str, 
        help="Folder save info of object features",
    )
    parser.add_argument(
        "--output_path",
        default="submission/submission_M4C_v0.json",
        type=str,
        help="The path obtain origin annotation",
    )
    return parser.parse_args()

if __name__ == "__main__":
    path = "/mlcv/Databases/VN_InfographicVQA/change/VietInfographicVQA_change_v1.0.json"
    DICT_QID = create_dic_question_id(path)    
    args = get_parser()
    main(args)

'''

python utils/convert_format_evaluation.py \
    --input_path="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/report/vi_infographicVQA/config_v0/vi_infographicvqa_run_test_2021-12-24T02:06:51.json" \
    --output_path="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/submission/ViInfographicVQA/submission_M4C_test_v0_VN.json"

python utils/convert_format_evaluation.py \
    --input_path="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/report/vi_infographicVQA/config_v0/vi_infographicvqa_run_val_2021-12-24T02:22:10.json" \
    --output_path="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/submission/ViInfographicVQA/submission_M4C_val_v0_VN.json"

python utils/convert_format_evaluation.py \
    --input_path="env_variable/report/vi_infographicVQA/config_v2/vi_infographicvqa_run_test_2021-12-25T02:26:08.json" \
    --output_path="submission/ViInfographicVQA/submission_M4C_test_v2_VN.json"

python utils/convert_format_evaluation.py \
    --input_path="env_variable/report/vi_infographicVQA/config_v2/vi_infographicvqa_run_val_2021-12-25T02:16:20.json" \
    --output_path="submission/ViInfographicVQA/submission_M4C_val_v2_VN.json"

python utils/convert_format_evaluation.py \
    --input_path="env_variable/report/inforgraphicVQA/config_v0/inforgraphicvqa_run_val_2021-09-29T16:25:54.json" \
    --output_path="submission/infographicVQA/submission_M4C_val_v0.json"


python utils/convert_format_evaluation.py \
    --input_path="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/report/vi_infographicVQA/config_v2/vi_infographicvqa_run_test_2021-12-25T02:26:08.json" \
    --output_path="submission/ViInfographicVQA/normal/submission_M4C_test_v2_VN.json"


python utils/convert_format_evaluation.py \
    --input_path="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/report/vi_infographicVQA/config_v5/vi_infographicvqa_run_val_2021-12-30T13:51:50.json" \
    --output_path="submission/ViInfographicVQA/normal/submission_M4C_val_v5_VN.json"
    
'''