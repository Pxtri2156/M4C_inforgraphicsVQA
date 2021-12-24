import os
import numpy as np
import random
import argparse
import json

def check_info_all(annotations):
    print("Element 0: ", annotations[0])
    print("Keys of 0 -> end: ", annotations[1].keys())
    print("The whole of annotations: ", annotations )
    # for annotation in annotations:
    #     print(type(annotation))
    #     print(json.dumps(annotation, indent=4))

def check_info_element(annotations):
    random_annotation = random.choice(annotations)
    print('keys: ', random_annotation.keys())
    print("Num key: ", len(random_annotation.keys()))
    for key in random_annotation.keys():

        if key != "ocr_info":
            print("{}: {}".format(key, random_annotation[key]))
        else:
            print("ocr_info")
            for e in random_annotation[key]:
                print("\t", e)


def check_question_equal_images(annotations):
    question_id = []
    image_id = []
    for annotation in annotations[1:]:
        question_id.append(annotation["question_id"])
        image_id.append(annotation['image_id'])
    question_id = set(question_id)
    image_id = set(image_id)
    print("number question id: ", len(question_id))
    print("number image id: ", len(image_id))
    if len(question_id) == len(image_id):
        return 1
    return 0

def find_question(annotations):
    questions_id = []
    images_id = []
    for annotation in annotations[1:]:
        question_id = annotation["question_id"]
        image_id = annotation['image_id']
        if image_id in images_id:
            print("image_id: {} question_id: {}".format(image_id, question_id))
        questions_id.append(question_id)
        images_id.append(image_id)

def show_question_require(annotations):
    for annotation in annotations[1:]:
        if annotation["question_id"] == 2 or annotation["question_id"] == 1:
            print(annotation)

def show_all_questions(annotations):
    print(len(annotations))
    for annotation in annotations[1:]:
        print(annotation["question_id"])

def show_all_ocr_tokens(annotations):
    for annotation in annotations[1:]:
        print(annotation["ocr_tokens"])

def show_all_ocr_tokens(annotations):
    for annotation in annotations[1:]:
        print(annotation["ocr_tokens"])

def find_img(annotations, img_name):
    for annotation in annotations[1:]:
        if annotation["image_name"]==img_name:
            print("Yes")
            return 1
    print("NO")

def find_num_max_answer(annotations):
    max_num = 0
    for annotation in annotations[1:]:
        if len(annotation["answers"]) > max_num:
            max_num = len(annotation["answers"])
    return max_num

def count_number_answer(annotations):
    for annotation in annotations[1:]:
        print(len(annotation["answers"]))

def show_all_answers(annotations):
    for annotation in annotations[1:]:
        print(annotation["answers"])

def check_number_answer(annotations, number_answer):
    for annotation in annotations[1:]:
        if len(annotation["answers"]) != number_answer:
            print(len(annotation["answers"]))
            print(annotation["answers"])

def sumarize_number_answer(annotations):
    sum = []
    for annotation in annotations[1:]:
        sum.append(len(annotation["answers"]))
    sum = set(sum)
    return sum

def check_question_id(annotations, question_id):
    for annotation in annotations[1:]:
        if annotation["question_id"] == question_id:
            return True
    return False

def count_type_answer(annotations):
    result = {1: 0}
    for annotation in annotations[1:]:
        num_answer = len(annotation["answers"])
        if num_answer not in result.keys():
            result[num_answer] = 1
        else:
            result[num_answer] += 1
    return result

def check_ocr_bbox(annotations):
    size_list = []
    for annotation in annotations[1:]:
        size_list.append(np.array(annotation['ocr_normalized_boxes']).shape)
        if np.array(annotation['ocr_normalized_boxes']).shape[0] == 0:
            print(annotation['image_name'])
            print(annotation['ocr_normalized_boxes'])
    size_list = set(size_list)
    print('size list: ', size_list)
    
def main(args):
    annotations = np.load(args.annot, allow_pickle=True) 
    print('len: ', len(annotations))
    if args.option == 0:
        print('Show all info of anotations')
        check_info_all(annotations)
    elif args.option == 1:
        print("Show info random element")
        check_info_element(annotations)
    elif args.option == 2:
        print("Is number question equal images ?")
        if check_question_equal_images(annotations) == 1:
            print("Yes")
        else:
            print("No")
    elif args.option == 3:
        print("Show image have one more question")
        find_question(annotations)
    elif args.option == 4:
        print("Show required question")
        show_question_require(annotations)
    elif args.option == 5:
        print("Show all question")
        show_all_questions(annotations)
    elif args.option == 6:
        print("Show all ocr tokens")
        show_all_ocr_tokens(annotations)
    elif args.option == 7:
        print("Find image")
        img_name="20467.jpeg"
        find_img(annotations, img_name)
    elif args.option == 8 :
        max_num_ans = find_num_max_answer(annotations)
        print("Max number answer: ", max_num_ans)
    elif args.option == 9:
        print("Show number answers")
        count_number_answer(annotations)
    elif args.option == 10:
        print("Show all answers")
        show_all_answers(annotations)
    elif args.option == 11:
        print("Check number answer")
        num_ans = 10
        check_number_answer(annotations, num_ans)
    elif args.option == 12:
        print("number answer")
        lst = sumarize_number_answer(annotations)
        print(lst)
    elif args.option == 13:
        print("No type answer")
        result = count_type_answer(annotations)
        print(result)
    elif args.option == 14: 
        question_id = 92270
        print("Check question id ", question_id)
        if check_question_id(annotations, question_id) == True:
            print("Yes")
        else:
            print("No")
    elif args.option == 15:
        check_ocr_bbox(annotations)
    else:
        print("Please choose suitable option")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annot", 
        default="env_variable/data/datasets/textvqa/defaults/annotations/imdb_train_ocr_en.npy", 
        type=str, 
        help="The folder obtain annotation of TextVQA"
    )
    parser.add_argument(
        "--option",
        default=0,
        type=int,
        help="Choose suitable option: "
        + "0: show all info of annnotation" 
        + "1: show info random element"
        + "2: check question equals image" 
        + "3: find image have one more question"
        + "4: Show required question"
        + "5: show all question"
        + "6: Show all ocr tokens", 
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    main(args)

'''
python utils/analysis_annotation.py \
    --annot="env_variable/data/datasets/textvqa/defaults/annotations/imdb_train_ocr_en.npy" \
    --option=1

python utils/analysis_annotation_textvqa.py \
    --annot="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/new_annotations/lmdb_val_en.npy" \
    --option=1



python utils/analysis_annotation.py \
    --annot="env_variable/data/datasets/inforgraphicvqa/defaults/annotations/infoVQA_train_en.npy" \
    --option=0

python utils/analysis_annotation.py \
--annot="env_variable/data/datasets/inforgraphicvqa/defaults/annotations/infoVQA_val_en.npy" \
--option=0

python utils/analysis_annotation.py \
--annot="env_variable/data/datasets/vi_infographicvqa/defaults/annotations/infoVQA_train_vi.npy" \
--option=0

'''