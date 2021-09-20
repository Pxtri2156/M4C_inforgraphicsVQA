import os
import argparse
import json 
import time



def gen_infor_dataset(data):
    result={
        "creation_time": time.time(),
        "version": data['dataset_version'],
        "dataset_type": data['dataset_split'],
        "has_answer": True
    }
    return result 
def get_info_ocr_result(ocr_results_path, name):
    ocr_path = os.path.join(ocr_results_path, name + ".json")
    ocr_file = open(ocr_path, )
    ocr_data = json.load(ocr_file)
    # print('ocr data: ', ocr_data)
    ocr_file.close()

def repair_lmdb_format(bboxes_folder, json_path, ocr_results_path, output_folder):
    annot_fi = open(json_path, )
    origin_annotation = json.load(annot_fi)
    print(origin_annotation.keys())
    info_dataset = gen_infor_dataset(origin_annotation)
    # print("info dataset", info_dataset)
    annots_data = origin_annotation['data']
    # print("annot data:", len(annots_data))
    for annot in annots_data:
        print(annot.keys())
        print(annot)
        #get info 
        ## question 
        question = annot['question']
        question_id = annot['questionId']
        question_tokens = question.strip("?").split()
        ## image 
        image_name = annot['image_local_name']
        image_id = image_name.replace(".jpeg", "")
        image_classes = []
        image_url = annot['image_url']
        img_width = 1
        img_height = 1 
        img_path= os.path.join(json_path.replace(".json", "_images"), image_name)
        ## answers
        feature_path = os.path.join(bboxes_folder, image_id + '.npy' )
        answers = annot['answers']
        valid_answers=answers
        ## ocr 
        get_info_ocr_result(ocr_results_path, image_id)
        ## another 
        set_name = annot['data_split']
        
        new_annotation={
            'question': question,
            'image_id': image_id,
            "image_classes": image_classes,
            "image_url": image_url, 
            "image_width": img_width,
            "image_height": img_height,
            "answers": answers,
            "question_tokens": question_tokens,
            "question_id": question_id,
            "set_name": set_name,
            "image_name": image_name,
            "image_path": img_path, 
            "feature_path": feature_path,
            "valid_answers": valid_answers,
            "ocr_tokens": " ", 
            "ocr_info": " ",
            "ocr_normalized_boxes": " ", 
            "obj_normalized_boxes": " "

        }
        # print('new annotation: ', new_annotation)
        break
    annot_fi.close()

def main(args):
    repair_lmdb_format(args.bboxes, args.origin_annot, args.ocr_results, args.output)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bboxes", 
        default="extracted_output", 
        type=str, 
        help="Folder save info of object features"
    )
    parser.add_argument(
        "--origin_annot",
        default="/mlcv/Databases/DocVQA_2020-21/task_3/val/infographicVQA_val_v1.0.json",
        type=str,
        help="The path obtain origin annotation",
    )
    parser.add_argument(
        "--ocr_results",
        default="/mlcv/Databases/DocVQA_2020-21/task_3/val/ocr_results",
        type=str,
        help="The folder save ocr results",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="The folder save news annotations",
        default="new_annotatons",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    main(args)

'''
python utils/repair_annotation.py \
    --bboxes="extracted_output" \
    --origin_annot="/mlcv/Databases/DocVQA_2020-21/task_3/val/infographicVQA_val_v1.0.json" \
    --ocr_results="/mlcv/Databases/DocVQA_2020-21/task_3/val/ocr_results" \
    --output="new_annotations"
'''