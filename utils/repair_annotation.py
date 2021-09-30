import os
import argparse
import json 
import time
import cv2
import numpy as np
import tqdm
import random

ERROR_IMGS = ['20467.jpeg']
MAX_NUM_ANS = 12

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
    ocr_tokens = []
    ocr_info = []
    ocr_normalized_boxes = []
    # print('keys ocr data: ', ocr_data.keys())
    # if "WORD" not in  ocr_data.keys():
    #     return ocr_tokens, ocr_info, ocr_normalized_boxes   
    word_data = ocr_data['WORD']
    for word in word_data:
        text = word['Text']
        ocr_tokens.append(text)
        bbox = word['Geometry']["BoundingBox"]
        polygon = word['Geometry']["Polygon"]
        bounding_box = {
            "topLeftX": bbox['Left'],
            "topLeftY": bbox['Top'],
            "width": bbox["Width"],
            "height": bbox["Height"],
            'rotation': 0,
             'roll': 0, 
             'pitch': 0, 
             'yaw': 0
        }
        info = {
            "word": text,
            "bounding_box": bounding_box,
        }
        ocr_info.append(info)
        ocr_normalized_box = [polygon[0]["X"], polygon[0]["Y"], polygon[2]["X"], polygon[2]["Y"]]
        ocr_normalized_boxes.append(ocr_normalized_box)
    # print("ocr_tokens", ocr_tokens)
    # print("ocr_info", ocr_info)
    # print("ocr_normalized_boxes ", ocr_normalized_boxes)
    ocr_file.close()
    return ocr_tokens, ocr_info, ocr_normalized_boxes

def get_obj_normalized_bboxes(feature_path):
    info_features = np.load(feature_path, allow_pickle=True) 
    info_features = info_features[()]
    obj_normalized_boxes = []
    width = info_features['image_width']
    height = info_features['image_height']
    for box in info_features["bbox"]:
        normalized_box = [box[0]/width, box[1]/height, box[2]/width, box[3]/height]
        obj_normalized_boxes.append(normalized_box)
    return obj_normalized_boxes

def save_annotation_results(results, output_folder):
    results_file = open(output_folder, "wb")
    results = np.array(results)
    np.save(results_file, results )
    results_file.close()
    print("Saved annotations results")

def gen_max_answer(answers, max_num_answer):
    # max_num_answer defined in config of data set
    cr_num_ans = len(answers)
    new_answers = []
    residual = max_num_answer % cr_num_ans
    n = max_num_answer // cr_num_ans
    for answer in answers:
        new_answers = new_answers + [answer]*n
    if residual != 0: 
        for i in range(residual):
            random_answer = random.choice(answers)
            new_answers.append(random_answer)
    return new_answers


def repair_textVQA_format(bboxes_folder, json_path, ocr_results_path, sub_num, output ):
    set_name = None
    annot_fi = open(json_path, )
    origin_annotation = json.load(annot_fi)
    m_annots_data = origin_annotation['data']
    # print("annot data:", len(annots_data))
    num_file = len(m_annots_data)
    distance = num_file//sub_num
    # print(origin_annotation.keys())
    info_dataset = gen_infor_dataset(origin_annotation)
    for k in range(0,num_file,distance):
        annotation_result = []
        annots_data = m_annots_data[k:k+distance]
        annotation_result.append(info_dataset)
    # print("info dataset", info_dataset)
        for i, annot in enumerate(annots_data):
            # print(annot.keys())
            # print(annot)
            #get info 
            ## question 
            question = annot['question']
            question_id = annot['questionId']
            question_tokens = question.strip("?").split()
            ## image 
            image_name = annot['image_local_name']
            print("Phrase {}/ {} processing [{}/{}]: {}".format(k/distance + 1,sub_num, i, distance, image_name))
            if image_name  in ERROR_IMGS:
                print("Skiped image:==================================== ", image_name)
                continue
            image_id = image_name.replace(".jpeg", "")
            image_classes = []
            image_url = annot['image_url']
            img_path= os.path.join(json_path.replace(".json", "_images"), image_name)
            img = cv2.imread(img_path)
            img_height, img_width, channels = img.shape
            set_name = annot['data_split']

            ## ocr 
            ocr_tokens, ocr_info, ocr_normalized_boxes =get_info_ocr_result(ocr_results_path, image_id)
            ## another 
            feature_path = os.path.join(bboxes_folder, image_id + '_info.npy' )
            # print('feature_path: ', feature_path)
            obj_normalized_boxes = get_obj_normalized_bboxes(feature_path)
            # feature_path = feature_path.split("/")[-1]
            new_annotation={
                'question': question,
                'image_id': image_id,
                "image_classes": image_classes,
                "image_url": image_url, 
                "image_width": img_width,
                "image_height": img_height,
                "question_tokens": question_tokens,
                "question_id": question_id,
                "set_name": set_name,
                "image_name": image_name,
                "image_path": img_path, 
                "feature_path": feature_path,
                "ocr_tokens": ocr_tokens, 
                "ocr_info": ocr_info,
                "ocr_normalized_boxes": ocr_normalized_boxes, 
                "obj_normalized_boxes": obj_normalized_boxes
            }
            ## answers
            if set_name != 'test':
                # print('before ', annot['answers'] )
                answers = gen_max_answer(annot['answers'], MAX_NUM_ANS)
                new_annotation['answers'] = answers
                # print('after ', answers )
                new_annotation['valid_answers'] = answers
            # print('new annotation: ', new_annotation)
            annotation_result.append(new_annotation)
            # input("Continue")
        if sub_num == 1:
            file_name = "infoVQA_{}_en.npy".format(set_name)
        else:
            file_name = "infoVQA_{}_en_{}.npy".format(set_name, str(k/distance))
        output_path = os.path.join(output, file_name)
        save_annotation_results(annotation_result, output_path)
    annot_fi.close()

def main(args):
    repair_textVQA_format(args.bboxes, args.origin_annot, args.ocr_results,args.sub_num, args.output)

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
    parser.add_argument(
        "--sub_num",
        type=int,
        help="The number of sub image for train: 3 or 6, test, valid: 1",
        default=3,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    main(args)

'''

python utils/repair_annotation.py \
    --bboxes="my_features/bbox_features/object" \
    --origin_annot="/mlcv/Databases/DocVQA_2020-21/task_3/train/infographicVQA_train_v1.0.json" \
    --ocr_results="/mlcv/Databases/DocVQA_2020-21/task_3/train/ocr_results" \
    --output="my_features/annotations/train" \
    --sub_num=1


python utils/repair_annotation.py \
    --bboxes="my_features/bbox_features/object" \
    --origin_annot="/mlcv/Databases/DocVQA_2020-21/task_3/train/infographicVQA_train_v1.0.json" \
    --ocr_results="/mlcv/Databases/DocVQA_2020-21/task_3/train/ocr_results" \
    --output="my_features/annotations/train" \
    --sub_num=3

python utils/repair_annotation.py \
    --bboxes="my_features/bbox_features/object" \
    --origin_annot="/mlcv/Databases/DocVQA_2020-21/task_3/val/infographicVQA_val_v1.0.json" \
    --ocr_results="/mlcv/Databases/DocVQA_2020-21/task_3/val/ocr_results" \
    --output="my_features/annotations/val" \
    --sub_num=1

python utils/repair_annotation.py \
    --bboxes="my_features/bbox_features/object" \
    --origin_annot="/mlcv/Databases/DocVQA_2020-21/task_3/test/infographicVQA_test_v1.0.json" \
    --ocr_results="/mlcv/Databases/DocVQA_2020-21/task_3/test/ocr_results" \
    --output="my_features/annotations/test" \
    --sub_num=1
'''