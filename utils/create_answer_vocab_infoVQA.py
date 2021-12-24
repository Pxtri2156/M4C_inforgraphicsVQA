import os
import argparse
import json

VOCABS_EXTENSION = [ '<pad>', '<s>', '</s>', '<unk>' ]

def get_vocab_from_annotations(input_path):
    vocabs = []
    fi = open(input_path)
    data = json.load(fi)
    annotations = data['data']
    for annotation in annotations:
        answers = annotation['answers']
        for answer in answers:
            answer_words = answer.split(" ")
            for word in answer_words:
                word = word.strip( ).strip(",")
                vocabs.append(word)
    fi.close()
    return vocabs

def get_vocab_train_test(train_path, val_path, save_path):
    print("Getting vocabs from train set")
    train_vocabs = get_vocab_from_annotations(train_path)
    print("Getting vocabs from val set")
    val_vocabs = get_vocab_from_annotations(val_path)
    vocabs =  train_vocabs + val_vocabs
    # print('Vocabs: ', vocabs)
    vocabs = set(vocabs)
    print("Writing answer vocabs")
    fi = open(save_path, 'w')
    for word in vocabs:
        fi.write(word + "\n")
    print("Saved answer vocabs")
    fi.close()

def main(args):
    get_vocab_train_test(args.train, args.val, args.output)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", 
        default="/mlcv/Databases/DocVQA_2020-21/task_3/train/infographicVQA_train_v1.0.json", 
        type=str, 
        help="Path save train annotations",
    )
    parser.add_argument(
        "--val", 
        default="/mlcv/Databases/DocVQA_2020-21/task_3/test/infographicVQA_test_v1.0.json", 
        type=str, 
        help="Path save valid annotations",
    )
    parser.add_argument(
        "--output",
        default="env_variable/data/datasets/textvqa/defaults/extras/vocabs/answer_vocab_infographic.txt",
        type=str,
        help="File path save answer vocab",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    main(args)

'''
python utils/get_answer_vocab_infoVQA.py \
    --train=/mlcv/Databases/DocVQA_2020-21/task_3/train/infographicVQA_train_v1.0.json \
    --val=/mlcv/Databases/DocVQA_2020-21/task_3/val/infographicVQA_val_v1.0.json \
    --output=env_variable/data/datasets/inforgraphicvqa/defaults/extras/vocabs/answer_vocab_infographic.txt
    

python utils/create_answer_vocab_infoVQA.py \
    --train=/mlcv/Databases/VN_InfographicVQA/train/VietInfographicVQA_train_v1.0.json \
    --val=/mlcv/Databases/VN_InfographicVQA/val/VietInfographicVQA_val_v1.0.json \
    --output=env_variable/data/datasets/vi_infographicvqa/defaults/vocabs/answer_vocab_ViInfographicVQA.txt
'''
