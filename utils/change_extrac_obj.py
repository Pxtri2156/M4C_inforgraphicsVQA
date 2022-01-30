import os
import argparse
import json 
import numpy as np
import tqdm
import sys
sys.path.append('./')
from utils.prepare_annotation import get_obj_normalized_bboxes, save_annotation_results

def change_obj_bbox(annotations_path, obj_bbox_path, output):
    annotations = np.load(annotations_path, allow_pickle=True)
    # annot_len = len(annotations)
    for annotation in annotations[1:]:
        img_id = annotation['image_id']
        print("Processing ", img_id)
        feature_info_path = os.path.join(obj_bbox_path, img_id + "_info.npy")
        obj_normalized_boxes = get_obj_normalized_bboxes(feature_info_path)
        annotation['obj_normalized_boxes'] = obj_normalized_boxes
        annotation['feature_path'] = feature_info_path
    save_annotation_results(annotations, output)
    print("Changed obj_norm_bboxes at ", output)

def main(args):
    change_obj_bbox(args.annotations, args.obj_bbox, args.output)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations", 
        default="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/data/datasets/inforgraphicvqa/defaults/annotations/infoVQA_train_en.npy", 
        type=str, 
        help="The path obtatin origin annotations",
    )
    parser.add_argument(
        "--obj_bbox", 
        default="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/data/datasets/inforgraphicvqa/20_object", 
        type=str, 
        help="Folder save new object bboxes",
    )
    parser.add_argument(
        "--output",
        default="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/data/datasets/inforgraphicvqa/30_object/annotations/infoVQA_train_en.npy",
        type=str,
        help="File path save new annotations",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    main(args)

'''
python utils/change_extrac_obj.py \
    --annotations="env_variable/data/datasets/inforgraphicvqa/defaults/annotations/infoVQA_test_en.npy" \
    --obj_bbox="env_variable/data/datasets/inforgraphicvqa/30_object/feature" \
    --output="env_variable/data/datasets/inforgraphicvqa/30_object/annotations/infoVQA_test_en.npy"


python utils/change_extrac_obj.py \
    --annotations="env_variable/data/datasets/vi_infographicvqa/defaults/annotations/infoVQA_test_vi.npy" \
    --obj_bbox="env_variable/data/datasets/vi_infographicvqa/100_object/features" \
    --output="env_variable/data/datasets/vi_infographicvqa/100_object/annotations/infoVQA_test_vi.npy"
 
'''