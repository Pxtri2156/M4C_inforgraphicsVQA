FOLDER_PATH_TO_DATASET="/mlcv/Databases/DocVQA_2020-21/task_3/val/infographicVQA_val_v1.0_images"
OUTPUT="lmdb_ocr"
IMDB_FILE="new_annotations/lmdb_val_en.npy"
DETECTION_MODEl="models_ocr/detectron_model.pth"
DETECTION_CONFIG="models_ocr/detectron_model.yaml"

CUDA_DEVICES_VISBLE=1 python projects/m4c/scripts/extract_ocr_frcn_feature.py \
    --image_dir=$FOLDER_PATH_TO_DATASET \
    --imdb_file=$IMDB_FILE \
    --detection_model=$DETECTION_MODEl \
    --detection_cfg=$DETECTION_CONFIG \
    --save_dir=$OUTPUT 