import os
import numpy as np
import argparse

def show_all_info(info_features):
    print("key of info feature: ", info_features.keys())
    print("No bbox: ", info_features["bbox"].shape)
    print("No object: ", info_features["objects"].shape)
    print("No cls_prob: ", info_features["cls_prob"].shape)
    print("feature: ",info_features)

def show_type_data(info_features):
    print(info_features)
    a, b = info_features.shape

def main(args):
    info_features = np.load(args.features, allow_pickle=True) 
    info_features = info_features[()]
    if args.option == 0:
        print("Show all info")
        show_all_info(info_features)
    elif args.option == 1: 
        show_type_data(info_features)
    else:
        print("Please choose option again!!!")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", 
        default="extracted_output", 
        type=str, 
        help="The folder obtain info of object bbox feature"
    )
    parser.add_argument(
        "--option",
        default=0,
        type=int,
        help="Choose suitable option: "
        + "0: show all info of annnotation of object feature", 
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    main(args)

'''
python utils/analysis_annotation_feature.py \
    --features="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/my_features/bbox_features/ocr/10002_info.npy" \
    --option=0
'''
