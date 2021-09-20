import os
import numpy as np
import random
import argparse

def check_info_all(annotations):
    print("Element 0: ", annotations[0])
    print("Keys of 0 -> end: ", annotations[1].keys())
    print("The whole of annotations: ", annotations)

def check_info_element(annotations):
    random_annotation = random.choice(annotations)
    print('keys: ', random_annotation.keys())
    for key in random_annotation.keys():
        print("{}: {}".format(key, random_annotation[key]))

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
        + "4: show all question", 
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    main(args)



'''
python utils/analysis_annotation_textvqa.py \
    --annot="env_variable/data/datasets/textvqa/defaults/annotations/imdb_train_ocr_en.npy" \
    --option=0
'''