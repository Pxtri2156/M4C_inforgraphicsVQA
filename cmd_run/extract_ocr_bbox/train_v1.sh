FOLDER_PATH_TO_DATASET="/mlcv/Databases/DocVQA_2020-21/task_3/train/infographicVQA_train_v1.0_images"
IMDB_FILE="my_features/annotations/train/infoVQA_train_en_1.0.npy"
DETECTION_MODEl="models_ocr/detectron_model.pth"
DETECTION_CONFIG="models_ocr/detectron_model.yaml"
OUTPUT="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/data/datasets/inforgraphicvqa/ocr_en/features"

CUDA_VISIBLE_DEVICES=7 python projects/m4c/scripts/extract_ocr_frcn_feature.py \
    --image_dir=$FOLDER_PATH_TO_DATASET \
    --imdb_file=$IMDB_FILE \
    --detection_model=$DETECTION_MODEl \
    --detection_cfg=$DETECTION_CONFIG \
    --save_dir=$OUTPUT 

