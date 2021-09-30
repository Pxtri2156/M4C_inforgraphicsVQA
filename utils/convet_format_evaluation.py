import json
import argparse
# input_path = "/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/report/inforgraphicVQA/inforgraphicvqa_run_val_2021-09-29T16:25:54.json"
# output_path = "/mlcv/Databases/DocVQA_2020-21/task_3/Task_3_evaluation_script/Task_3/submissions/val_inforgraphic.json"

def convert_format_submission(input_path, output_path):
    fi = open(input_path, )
    out_fi = open(output_path, 'w')
    data = json.load(fi)
    new_data = []
    print(len(data))
    for element in data:
        print("Processing question ",  element['question_id'] )
        # print(type(element))
        new_data.append({
            "questionId": element['question_id'],
            "answer": element['answer']
        })
    print("Done")
    json.dump(new_data, out_fi)
    fi.close()
    out_fi.close()

def main(args):
    convert_format_submission(args.input_path, args.output_path)

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
    args = get_parser()
    main(args)