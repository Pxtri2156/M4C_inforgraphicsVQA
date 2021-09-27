OBJECT_FEATURES="my_features/bbox_features/object"
ORIGIN_ANNOT="/mlcv/Databases/DocVQA_2020-21/task_3/train/infographicVQA_train_v1.0.json"
OCR_RESULTS="/mlcv/Databases/DocVQA_2020-21/task_3/train/ocr_results"
OUTPUT="env_variable/data/datasets/inforgraphicvqa/defaults/annotations/vqa_train_en.npy"

python utils/repair_annotation.py \
    --bboxes=$OBJECT_FEATURES \
    --origin_annot=$ORIGIN_ANNOT \
    --ocr_results=$OCR_RESULTS \
    --output=$OUTPUT