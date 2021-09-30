FOLDER_PATH_TO_DATASET="/mlcv/Databases/DocVQA_2020-21/task_3/test/infographicVQA_test_v1.0_images"
IMDB_FILE="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/my_features/annotations/test/infoVQA_test_en.npy"
DETECTION_MODEl="models_ocr/detectron_model.pth"
DETECTION_CONFIG="models_ocr/detectron_model.yaml"
OUTPUT="/mlcv/WorkingSpace/NCKH/tiennv/vqa_thesis/docvqa/libs/mmf/env_variable/data/datasets/inforgraphicvqa/ocr_en/features"

CUDA_VISIBLE_DEVICES=7 python projects/m4c/scripts/extract_ocr_frcn_feature.py \
    --image_dir=$FOLDER_PATH_TO_DATASET \
    --imdb_file=$IMDB_FILE \
    --detection_model=$DETECTION_MODEl \
    --detection_cfg=$DETECTION_CONFIG \
    --save_dir=$OUTPUT 